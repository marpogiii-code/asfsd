import os
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import gradio as gr
import spaces
import torch
from fastapi.responses import FileResponse, HTMLResponse
from gradio import processing_utils
from gradio.utils import abspath, get_upload_folder, is_in_or_equal
from transformers import AutoModelForMultimodalLM, AutoProcessor, BatchFeature, StoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer

from params import PARAM_SPECS, inject_param_config, validate_params

MODEL_ID = "google/gemma-4-12b-it"

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16)

IMAGE_FILE_TYPES = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_FILE_TYPES = (".wav", ".mp3", ".flac", ".ogg")
VIDEO_FILE_TYPES = (".mp4", ".mov", ".avi", ".webm")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "10_000"))

THINKING_START = "<|channel>"
THINKING_END = "<channel|>"

STATIC_DIR = Path(__file__).parent / "static"

# Special tokens to strip from decoded output (keeping thinking delimiters
# so that the reasoning section can be split out below).
_KEEP_TOKENS = {THINKING_START, THINKING_END}
_STRIP_TOKENS = sorted(
    (t for t in processor.tokenizer.all_special_tokens if t not in _KEEP_TOKENS),
    key=len,
    reverse=True,  # longest first to avoid partial matches
)


def _strip_special_tokens(text: str) -> str:
    for tok in _STRIP_TOKENS:
        text = text.replace(tok, "")
    return text


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split accumulated thinking-mode output into (reasoning, content).

    The model only emits a reasoning channel when it actually reasons; a direct
    answer has no delimiters. So text is reasoning only while it starts with the
    opening delimiter, mirroring Gradio's ``reasoning_tags`` semantics.
    """
    if not text.startswith(THINKING_START):
        return "", text
    body = text[len(THINKING_START) :].removeprefix("thought\n")
    if THINKING_END in body:
        reasoning, content = body.split(THINKING_END, 1)
        return reasoning, content
    return body, ""  # reasoning channel still streaming


def _classify_file(path: str) -> str | None:
    """Return media type string for a file path, or None if unsupported."""
    lower = path.lower()
    if lower.endswith(IMAGE_FILE_TYPES):
        return "image"
    if lower.endswith(AUDIO_FILE_TYPES):
        return "audio"
    if lower.endswith(VIDEO_FILE_TYPES):
        return "video"
    return None


def _resolve_media_source(path: str) -> str:
    """Resolve a client-supplied media reference to a safe local path.

    The chat endpoint takes raw path/URL strings, so unlike a normal Gradio
    component it does not get Gradio's built-in input guards for free. Mirror
    them here by reusing Gradio's own helpers: download remote URLs through the
    SSRF-guarded path (which rejects private/link-local hosts and re-checks
    redirects) and restrict local paths to files actually uploaded via
    /gradio_api/upload. Otherwise the processor would read arbitrary server
    paths and fetch arbitrary URLs on the client's behalf.
    """
    upload_folder = get_upload_folder()
    if path.startswith(("http://", "https://")):
        return processing_utils.ssrf_protected_download(path, cache_dir=upload_folder)
    if not is_in_or_equal(path, upload_folder):
        raise gr.Error("Invalid file path.")
    return str(abspath(path))


def _user_content(text: str, files: list[str]) -> list[dict]:
    """Build a user message content list from text and uploaded file paths.

    Media placement follows the model's guidance: image and video go before the
    text, audio goes after it. Putting audio after the text noticeably helps the
    12B model stay on task (placing it first makes it more prone to degenerate,
    looping output).
    """
    before: list[dict] = []
    after: list[dict] = []
    for path in files:
        kind = _classify_file(path)
        if kind == "audio":
            after.append({"type": kind, "url": _resolve_media_source(path)})
        elif kind:
            before.append({"type": kind, "url": _resolve_media_source(path)})
    return [*before, {"type": "text", "text": text}, *after]


def process_history(history: list[dict]) -> list[dict]:
    """Convert the frontend chat history into chat-template messages.

    Each history item is ``{"role", "text", "files"}``. Assistant reasoning is
    not stored on the frontend, so only the final answer is fed back.
    """
    messages: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["text"]}]})
        else:
            messages.append({"role": "user", "content": _user_content(item["text"], item.get("files", []))})
    return messages


class StopOnSignal(StoppingCriteria):
    def __init__(self) -> None:
        self.stopped = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: object) -> bool:  # noqa: ARG002
        return self.stopped


@spaces.GPU(duration=120)
def _generate_on_gpu(
    inputs: BatchFeature,
    max_new_tokens: int,
    thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[str]:
    inputs = inputs.to(device=model.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(
        processor,
        timeout=30.0,
        skip_prompt=True,
        skip_special_tokens=not thinking,
    )
    stop_criteria = StopOnSignal()
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "stopping_criteria": [stop_criteria],
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "disable_compile": True,
    }
    if temperature > 0:
        # Sampling (the model's default). Temperature 0 means greedy decoding.
        generate_kwargs |= {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    else:
        generate_kwargs["do_sample"] = False

    exception_holder: list[Exception] = []

    def _generate() -> None:
        try:
            model.generate(**generate_kwargs)
        except Exception as e:  # noqa: BLE001
            exception_holder.append(e)
        finally:
            # generate() only signals the streamer on the normal path, so a
            # failure (CUDA OOM, etc.) would otherwise leave the consumer
            # blocked until the timeout, masking the real error with a
            # queue.Empty. End it here so the loop returns and exception_holder
            # is surfaced below.
            streamer.end()

    thread = Thread(target=_generate)
    thread.start()

    chunks: list[str] = []
    try:
        for text in streamer:
            chunks.append(text)
            accumulated = "".join(chunks)
            if thinking:
                yield _strip_special_tokens(accumulated)
            else:
                yield accumulated
    finally:
        # Stop generation and reclaim the worker thread on every exit path:
        # normal completion, client disconnect (GeneratorExit), and a streamer
        # timeout (queue.Empty). The text queue is unbounded, so generate()
        # never blocks on put; signalling the stop criteria lets it return at
        # the next token, after which the join completes.
        stop_criteria.stopped = True
        thread.join()

    if exception_holder:
        msg = f"Generation failed: {exception_holder[0]}"
        raise gr.Error(msg)


def _validate(text: str, files: list[str]) -> None:
    if not text.strip() and not files:
        raise gr.Error("Please enter a message or upload a file.")

    kinds = [k for k in (_classify_file(f) for f in files) if k is not None]
    if len(set(kinds)) > 1:
        raise gr.Error("Please upload only one type of media (images, audio, or video) at a time.")
    if kinds.count("audio") > 1:
        raise gr.Error("Only one audio file can be uploaded at a time.")
    if kinds.count("video") > 1:
        raise gr.Error("Only one video file can be uploaded at a time.")


app = gr.Server()


@app.api(name="chat")
def chat(
    text: str,
    files: list[str] | None = None,
    history: list[dict] | None = None,
    thinking: bool = False,
    max_new_tokens: int = int(PARAM_SPECS["max_new_tokens"].default),
    image_token_budget: int = int(PARAM_SPECS["image_token_budget"].default),
    system_prompt: str = "",
    temperature: float = PARAM_SPECS["temperature"].default,
    top_p: float = PARAM_SPECS["top_p"].default,
    top_k: int = int(PARAM_SPECS["top_k"].default),
    repetition_penalty: float = PARAM_SPECS["repetition_penalty"].default,
) -> Iterator[dict]:
    """Stream a Gemma response as ``{"reasoning", "content"}`` updates.

    Args:
        text: The new user message.
        files: Server-side paths of files uploaded via /gradio_api/upload.
        history: Prior turns as a list of {"role", "text", "files"}.
        thinking: Whether to enable the model's reasoning channel.
        max_new_tokens: Maximum number of tokens to generate.
        image_token_budget: Soft cap on image tokens (higher preserves detail).
        system_prompt: Optional system prompt.
        temperature: Sampling temperature; 0 means greedy decoding.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalty for repeated tokens (1.0 disables it).
    """
    files = files or []
    history = history or []

    _validate(text, files)
    params = validate_params(
        {
            "max_new_tokens": max_new_tokens,
            "image_token_budget": image_token_budget,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
    )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.extend(process_history(history))
    messages.append({"role": "user", "content": _user_content(text, files)})

    has_video = any(c.get("type") == "video" for m in messages for c in m["content"])
    template_kwargs: dict = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "load_audio_from_video": has_video,
        "processor_kwargs": {"images_kwargs": {"max_soft_tokens": params["image_token_budget"]}},
    }
    if thinking:
        template_kwargs["enable_thinking"] = True

    inputs = processor.apply_chat_template(messages, **template_kwargs)

    n_tokens = inputs["input_ids"].shape[1]
    if n_tokens > MAX_INPUT_TOKENS:
        msg = f"Input too long ({n_tokens} tokens). Maximum is {MAX_INPUT_TOKENS} tokens."
        raise gr.Error(msg)

    for raw in _generate_on_gpu(
        inputs=inputs,
        max_new_tokens=params["max_new_tokens"],
        thinking=thinking,
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        repetition_penalty=params["repetition_penalty"],
    ):
        if thinking:
            reasoning, content = _split_reasoning(raw)
        else:
            reasoning, content = "", raw
        yield {"reasoning": reasoning, "content": content}


# Serve the frontend with no-store so a reload always picks up the latest
# build; stale cached assets mixed with new ones break the layout.
_NO_STORE = {"Cache-Control": "no-store"}


@app.get("/")
def index() -> HTMLResponse:
    # Inject the parameter specs so the UI controls and the chat API share one
    # definition (params.py). Read on each request so a reload picks up the
    # latest build, matching the no-store policy.
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(inject_param_config(html), headers=_NO_STORE)


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "app.js", media_type="text/javascript", headers=_NO_STORE)


@app.get("/style.css")
def style_css() -> FileResponse:
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css", headers=_NO_STORE)


if __name__ == "__main__":
    app.launch(allowed_paths=[str(STATIC_DIR)], max_file_size="20MB")
