import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";
import { marked } from "https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js";
import DOMPurify from "https://cdn.jsdelivr.net/npm/dompurify/dist/purify.es.mjs";
import renderMathInElement from "https://cdn.jsdelivr.net/npm/katex/dist/contrib/auto-render.mjs";
import hljs from "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/es/highlight.min.js";

const banner = document.getElementById("banner");
const bannerText = document.getElementById("banner-text");

let bannerTimer;
function hideBanner() {
  banner.hidden = true;
  clearTimeout(bannerTimer);
}

document.getElementById("banner-close").addEventListener("click", hideBanner);

function showBanner(message, { autoHide = false, notice = false } = {}) {
  bannerText.textContent = message;
  banner.classList.toggle("notice", notice);
  banner.hidden = false;
  clearTimeout(bannerTimer);
  if (autoHide) {
    bannerTimer = setTimeout(hideBanner, 4000);
  }
}

let client = null;
try {
  // Subscribe to "status" in addition to "data": the client only publishes the
  // event types listed here to the submit() async iterator, and a raised
  // gr.Error arrives as a {type:"status", stage:"error"} event. Without "status"
  // the error never reaches the for-await loop and the failure is swallowed.
  client = await Client.connect(location.origin, { events: ["data", "status"] });
} catch {
  showBanner("Could not connect to the server. Please reload the page.");
}

const chatScroll = document.getElementById("chat-scroll");
const chatLog = document.getElementById("chat-log");
const messageBox = document.getElementById("message");
const sendBtn = document.getElementById("send-btn");
const thinkingToggle = document.getElementById("thinking");
const fileInput = document.getElementById("file-input");
const attachBtn = document.getElementById("attach-btn");
const fileChips = document.getElementById("file-chips");
const systemPromptBox = document.getElementById("system-prompt");
const maxTokensInput = document.getElementById("max-tokens");
const maxTokensOut = document.getElementById("max-tokens-out");
const imageBudgetSelect = document.getElementById("image-budget");
const temperatureInput = document.getElementById("temperature");
const topPInput = document.getElementById("top-p");
const topKInput = document.getElementById("top-k");
const repPenaltyInput = document.getElementById("rep-penalty");
const dropOverlay = document.getElementById("drop-overlay");
const lightbox = document.getElementById("lightbox");
const newChatBtn = document.getElementById("new-chat-btn");
const mobileNewChatBtn = document.getElementById("mobile-new-chat-btn");
const app = document.querySelector(".app");
const menuBtn = document.getElementById("menu-btn");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const editLastBtn = document.getElementById("edit-last-btn");

const ACCEPTED_EXTENSIONS = [
  ".jpg", ".jpeg", ".png", ".webp",
  ".wav", ".mp3", ".flac", ".ogg",
  ".mp4", ".mov", ".avi", ".webm",
];

function isAccepted(file) {
  const name = file.name.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => name.endsWith(ext));
}

// Media kind from a MIME type (local files) or a URL extension (examples).
function typeKind(mime) {
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("audio/")) return "audio";
  return "image";
}

function extKind(url) {
  const ext = url.split("?")[0].split(".").pop().toLowerCase();
  if (["mp4", "mov", "avi", "webm"].includes(ext)) return "video";
  if (["mp3", "wav", "flac", "ogg"].includes(ext)) return "audio";
  return "image";
}

// Free any object URLs an attachment holds (remote-URL items have none).
function revokeItem(item) {
  if (item.file) URL.revokeObjectURL(item.url);
}

// Attachments are either a local File (uploaded on send) or a remote URL
// (passed straight to the model, e.g. example media). Both carry { kind, url }
// for display; local items add `file`, remote items add `remoteUrl`.
function addFiles(fileList) {
  const all = Array.from(fileList);
  const accepted = all.filter(isAccepted);
  for (const file of accepted) {
    pendingFiles.push({ file, url: URL.createObjectURL(file), kind: typeKind(file.type) });
  }
  if (accepted.length) renderPreviews();
  const rejected = all.length - accepted.length;
  if (rejected > 0) {
    const noun = rejected === 1 ? "file" : "files";
    showBanner(`Skipped ${rejected} unsupported ${noun}.`, { autoHide: true, notice: true });
  }
}

// Conversation history sent back to the backend: {role, text, files}.
const history = [];
// Files attached to the next message, not yet uploaded: {file, url}.
let pendingFiles = [];
let currentJob = null;
// Set when the user presses Stop, so the turn's end can be told apart from a
// failure (a deliberate stop is silent; a failure gets a banner).
let cancelled = false;
// The most recent sent turn, kept so it can be taken back into the composer.
let lastTurn = null;
// Bumped whenever a new turn starts or the conversation is reset. A streaming
// turn captures this value and bails out of its tail work if it no longer
// matches, so a "Clear chat" mid-stream can't resurrect the discarded turn.
let turnEpoch = 0;

maxTokensInput.addEventListener("input", () => {
  maxTokensOut.textContent = maxTokensInput.value;
});

// Mirror each slider's value into its adjacent <output>.
for (const id of ["temperature", "top-p", "top-k", "rep-penalty"]) {
  const input = document.getElementById(id);
  const out = document.getElementById(`${id}-out`);
  input.addEventListener("input", () => {
    out.textContent = input.value;
  });
}

// Backend parameter -> its control element ids. The allowed ranges, steps, and
// defaults are applied from window.PARAM_CONFIG (sourced from params.py), so
// they are defined in one place rather than duplicated in this file.
const PARAM_CONTROLS = [
  { key: "max_new_tokens", input: "max-tokens", out: "max-tokens-out" },
  { key: "image_token_budget", input: "image-budget" },
  { key: "temperature", input: "temperature", out: "temperature-out" },
  { key: "top_p", input: "top-p", out: "top-p-out" },
  { key: "top_k", input: "top-k", out: "top-k-out" },
  { key: "repetition_penalty", input: "rep-penalty", out: "rep-penalty-out" },
];

function applyParamConfig() {
  const config = window.PARAM_CONFIG ?? {};
  for (const { key, input, out } of PARAM_CONTROLS) {
    const spec = config[key];
    if (!spec) continue;
    const el = document.getElementById(input);
    if (spec.choices) {
      // Discrete select: build its options from the spec.
      el.replaceChildren(
        ...spec.choices.map((choice) => {
          const opt = document.createElement("option");
          opt.value = String(choice);
          opt.textContent = String(choice);
          opt.selected = choice === spec.default;
          return opt;
        }),
      );
    } else {
      // Range slider.
      el.min = spec.min;
      el.max = spec.max;
      el.step = spec.step;
      el.value = spec.default;
    }
    if (out) document.getElementById(out).textContent = spec.default;
  }
}

applyParamConfig();

messageBox.addEventListener("input", () => {
  messageBox.style.height = "auto";
  messageBox.style.height = `${messageBox.scrollHeight}px`;
});

messageBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});

// Drag-and-drop. A counter avoids flicker when dragging over child elements.
let dragDepth = 0;

function hasFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}

window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth += 1;
  dropOverlay.hidden = false;
});

window.addEventListener("dragover", (e) => {
  if (hasFiles(e)) e.preventDefault();
});

window.addEventListener("dragleave", (e) => {
  if (!hasFiles(e)) return;
  dragDepth -= 1;
  if (dragDepth <= 0) {
    dragDepth = 0;
    dropOverlay.hidden = true;
  }
});

window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.hidden = true;
  addFiles(e.dataTransfer.files);
});

// Paste images straight from the clipboard.
window.addEventListener("paste", (e) => {
  const files = Array.from(e.clipboardData?.files || []);
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
});

// Lightbox: click a chat image to view it full size.
function closeLightbox() {
  lightbox.hidden = true;
  lightbox.innerHTML = "";
}

function openLightbox(kind, url) {
  lightbox.innerHTML = "";
  let el;
  if (kind === "audio") {
    el = document.createElement("audio");
    el.controls = true;
    el.autoplay = true;
  } else if (kind === "video") {
    el = document.createElement("video");
    el.controls = true;
    el.autoplay = true;
    // Play inline; without this, mobile WebKit forces native fullscreen, which
    // shifts the video out of place and leaves the page scrollable afterwards.
    el.playsInline = true;
    el.setAttribute("playsinline", "");
    // Don't close on the media itself; let clicks use the native controls.
  } else {
    el = document.createElement("img");
    el.addEventListener("click", closeLightbox);
  }
  el.src = url;
  lightbox.appendChild(el);

  // The media can fill the screen, making the backdrop hard to tap, so give it
  // an explicit close button.
  const close = document.createElement("button");
  close.className = "lightbox-close";
  close.type = "button";
  close.setAttribute("aria-label", "Close");
  close.textContent = "✕";
  close.addEventListener("click", closeLightbox);
  lightbox.appendChild(close);

  lightbox.hidden = false;
}

// Clicking the backdrop (outside the media) closes it; so does Escape.
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});
// Escape closes the lightbox first (it's modal); otherwise it dismisses the banner.
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!lightbox.hidden) {
    closeLightbox();
  } else if (!banner.hidden) {
    hideBanner();
  }
});

// Example media is referenced by remote URL (not bundled): the browser shows it
// with <img>/<video> and the model fetches the URL server-side. This avoids
// redistributing copyrighted media in the repo.
//
// Order is intentional: text examples run roughly easiest-first, then the
// multimodal ones. This order is preserved on every screen. The `mobile: true`
// flag marks a small one-per-modality subset (text, image, audio, video) kept
// visible on narrow screens; the rest collapse behind "show more" there. The
// flag does not affect ordering, only which entries show before expanding.
const EXAMPLES = [
  { text: "What is the capital of France?", mobile: true },
  { text: "What is the water formula?" },
  { text: "Explain quantum entanglement in simple terms." },
  { text: "I want to do a car wash that is 50 meters away, should I walk or drive?" },
  {
    text: "Write a poem about beer with 4 stanzas. Format the title as an H2 markdown heading and bold the first line of each stanza.",
  },
  {
    text: "Describe this image.",
    files: ["https://news.bbc.co.uk/media/images/38107000/jpg/_38107299_ronaldogoal_ap_300.jpg"],
    mobile: true,
  },
  {
    text: "What are the key similarities between these three images?",
    files: [
      "https://news.bbc.co.uk/media/images/38107000/jpg/_38107299_ronaldogoal_ap_300.jpg",
      "https://ogimg.infoglobo.com.br/in/12547538-502-0e0/FT1086A/94-8705-14.jpg",
      "https://amazonasatual.com.br/wp-content/uploads/2021/01/Pele.jpg",
    ],
  },
  {
    text: "Transcribe the audio.",
    files: ["https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"],
    mobile: true,
  },
  {
    text: "Translate to Dutch.",
    files: ["https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"],
  },
  {
    text: "What is happening in this video?",
    files: ["https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"],
    mobile: true,
  },
];

// Fill the composer from an example. The user still presses send (so they can
// edit first). Example media are remote URLs, attached without uploading.
function useExample(ex) {
  messageBox.value = ex.text;
  autoResizeMessage();
  // Clicking an example replaces the composer, so drop any current attachments.
  pendingFiles.forEach(revokeItem);
  pendingFiles = (ex.files || []).map((url) => ({ url, remoteUrl: url, kind: extKind(url) }));
  renderPreviews();
  messageBox.focus();
}

// A small preview for a media example. Images show a thumbnail; audio and video
// use an icon (so the page doesn't download large media just for a thumbnail).
// A count badge marks examples that attach more than one file.
function exampleThumb(files) {
  const wrap = document.createElement("span");
  wrap.className = "example-media";

  const kind = extKind(files[0]);
  let thumb;
  if (kind === "image") {
    thumb = document.createElement("img");
    thumb.className = "example-thumb";
    thumb.src = files[0];
  } else {
    thumb = document.createElement("span");
    thumb.className = "example-icon";
    thumb.textContent = kind === "video" ? "🎬" : "🎵";
  }
  wrap.appendChild(thumb);

  if (files.length > 1) {
    const badge = document.createElement("span");
    badge.className = "example-count";
    badge.textContent = String(files.length);
    wrap.appendChild(badge);
  }
  return wrap;
}

function showEmptyState() {
  chatLog.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";

  const hint = document.createElement("p");
  hint.className = "empty-hint";
  hint.textContent = "Start the conversation below, or try an example:";
  empty.appendChild(hint);

  const examples = document.createElement("div");
  examples.className = "examples";
  for (const ex of EXAMPLES) {
    const btn = document.createElement("button");
    btn.className = "example";
    // Entries without the mobile flag collapse on narrow screens (see CSS).
    if (!ex.mobile) {
      btn.classList.add("secondary");
    }
    if (ex.files?.length) {
      btn.appendChild(exampleThumb(ex.files));
    }
    const label = document.createElement("span");
    label.textContent = ex.text;
    btn.appendChild(label);
    btn.addEventListener("click", () => useExample(ex));
    examples.appendChild(btn);
  }
  empty.appendChild(examples);

  // On a narrow screen, CSS hides the .secondary examples; this button reveals
  // them. It is hidden on desktop (and once expanded) by CSS, so the extra
  // entries always show there. Clicking expands one-way: reveal all, then hide
  // the button itself.
  const hidden = EXAMPLES.filter((ex) => !ex.mobile).length;
  if (hidden > 0) {
    const showMore = document.createElement("button");
    showMore.className = "show-more";
    showMore.textContent = `Show ${hidden} more examples`;
    showMore.addEventListener("click", () => {
      examples.classList.add("expanded");
      showMore.remove();
    });
    empty.appendChild(showMore);
  }

  chatLog.appendChild(empty);
}

function autoResizeMessage() {
  messageBox.style.height = "auto";
  messageBox.style.height = `${messageBox.scrollHeight}px`;
}

// Reset to a fresh conversation without reloading the page.
function resetConversation() {
  // Invalidate any in-flight turn so its tail work (after the upload/stream
  // awaits) is skipped instead of writing into the now-cleared conversation.
  turnEpoch += 1;
  if (currentJob) currentJob.cancel();
  setGenerating(false);
  history.length = 0;
  pendingFiles.forEach(revokeItem);
  pendingFiles = [];
  renderPreviews();
  messageBox.value = "";
  messageBox.style.height = "auto";
  lastTurn = null;
  editLastBtn.hidden = true;
  showEmptyState();
  app.classList.remove("sidebar-open"); // close the drawer on mobile
}

newChatBtn.addEventListener("click", resetConversation);
mobileNewChatBtn.addEventListener("click", resetConversation);

// Remove a turn's bubbles from the chat log, bringing back the empty state if
// the log is left empty. Shared by the take-back button and the no-answer path.
function removeTurn(userEl, assistantEl) {
  userEl?.remove();
  assistantEl?.remove();
  if (!chatLog.querySelector(".msg")) showEmptyState();
}

// Drop a turn's text and attachments back into the composer for editing.
function loadComposer(text, items) {
  messageBox.value = text;
  pendingFiles = items;
  renderPreviews();
  autoResizeMessage();
  messageBox.focus();
}

// Take the last sent turn back into the composer to edit and resend.
function editLastTurn() {
  if (!lastTurn) return;
  history.length = lastTurn.historyLenBefore;
  removeTurn(lastTurn.userEl, lastTurn.assistantEl);
  loadComposer(lastTurn.text, lastTurn.items);
  lastTurn = null;
  editLastBtn.hidden = true;
}

editLastBtn.addEventListener("click", editLastTurn);

// Mobile sidebar drawer
menuBtn.addEventListener("click", () => app.classList.add("sidebar-open"));
sidebarBackdrop.addEventListener("click", () => app.classList.remove("sidebar-open"));

function renderPreviews() {
  fileChips.innerHTML = "";
  pendingFiles.forEach((item, i) => {
    const preview = document.createElement("div");
    preview.className = "preview";

    // Clicking the thumbnail opens the file in the lightbox so it can be
    // checked before sending.
    if (item.kind === "image") {
      const img = document.createElement("img");
      img.className = "preview-thumb";
      img.src = item.url;
      img.addEventListener("click", () => openLightbox("image", item.url));
      preview.appendChild(img);
    } else if (item.kind === "video") {
      const video = document.createElement("video");
      video.className = "preview-thumb";
      video.src = item.url;
      video.addEventListener("click", () => openLightbox("video", item.url));
      preview.appendChild(video);
    } else {
      preview.classList.add("preview-file");
      const name = item.file ? item.file.name : item.remoteUrl.split("/").pop().split("?")[0];
      const label = document.createElement("span");
      label.textContent = `🎵 ${name}`;
      label.addEventListener("click", () => openLightbox("audio", item.url));
      preview.appendChild(label);
    }

    const remove = document.createElement("button");
    remove.className = "preview-remove";
    remove.textContent = "✕";
    remove.addEventListener("click", () => {
      revokeItem(item);
      pendingFiles.splice(i, 1);
      renderPreviews();
    });
    preview.appendChild(remove);
    fileChips.appendChild(preview);
  });
}

function clearEmptyState() {
  chatLog.querySelector(".empty-state")?.remove();
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function isNearBottom() {
  return chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight < 80;
}

const MATH_DELIMITERS = [
  { left: "$$", right: "$$", display: true },
  { left: "$", right: "$", display: false },
];

// Markdown often uses \[ \] and \( \) for math, but a CommonMark renderer
// strips those backslashes, so normalize them to $$ and $ before parsing.
function preprocessMath(text) {
  return text
    .replace(/\\\[/g, "$$$$")
    .replace(/\\\]/g, "$$$$")
    .replace(/\\\(/g, "$$")
    .replace(/\\\)/g, "$$");
}

// Render assistant markdown: parse, sanitize (defends against HTML the model
// might emit), then typeset math in place.
function renderAssistant(bubble, content) {
  bubble.innerHTML = DOMPurify.sanitize(marked.parse(preprocessMath(content)));
  // Open links in a new tab. This app is volatile, so following a link in the
  // same tab would throw away the whole conversation with no way back.
  bubble.querySelectorAll("a[href]").forEach((a) => {
    a.target = "_blank";
    a.rel = "noopener noreferrer";
  });
  renderMathInElement(bubble, { delimiters: MATH_DELIMITERS, throwOnError: false });
}

function addCopyButtons(container) {
  container.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement?.classList.contains("code-wrap")) return;
    // Wrap the scrolling <pre> in a non-scrolling container so the button can
    // stay pinned to the top-right instead of scrolling with the code.
    const wrap = document.createElement("div");
    wrap.className = "code-wrap";
    pre.replaceWith(wrap);
    wrap.appendChild(pre);
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", () => {
      const code = pre.querySelector("code")?.textContent ?? pre.textContent;
      navigator.clipboard.writeText(code);
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1500);
    });
    wrap.appendChild(btn);
  });
}

// Syntax-highlight code blocks that the model tagged with a language
// (```python etc.). Untagged blocks are left plain — no auto-detection.
function highlightCode(container) {
  container.querySelectorAll('pre code[class*="language-"]').forEach((block) => {
    if (block.dataset.highlighted) return;
    hljs.highlightElement(block);
  });
}

function addUserMessage(text, items) {
  clearEmptyState();
  const msg = document.createElement("div");
  msg.className = "msg user";
  for (const { kind, url } of items) {
    if (kind === "image") {
      const img = document.createElement("img");
      img.className = "msg-media";
      img.src = url;
      img.addEventListener("click", () => openLightbox("image", url));
      msg.appendChild(img);
    } else if (kind === "video") {
      const video = document.createElement("video");
      video.className = "msg-media";
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      video.setAttribute("playsinline", "");
      video.addEventListener("click", () => openLightbox("video", url));
      msg.appendChild(video);
    } else {
      const audio = document.createElement("audio");
      audio.src = url;
      audio.controls = true;
      msg.appendChild(audio);
    }
  }
  if (text) {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    msg.appendChild(bubble);
  }
  chatLog.appendChild(msg);
  scrollToBottom();
  return msg;
}

function addAssistantMessage() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";

  const reasoning = document.createElement("details");
  reasoning.className = "reasoning";
  reasoning.open = true;
  reasoning.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";
  const reasoningBody = document.createElement("div");
  reasoningBody.className = "reasoning-body md";
  reasoning.append(summary, reasoningBody);

  const bubble = document.createElement("div");
  bubble.className = "bubble md";
  bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';

  msg.append(reasoning, bubble);
  chatLog.appendChild(msg);
  scrollToBottom();
  return { msg, reasoning, reasoningBody, bubble };
}

// Append a short note under an assistant turn. `kind` is "stopped" or "error".
function addTurnNote(msg, text, kind) {
  const note = document.createElement("div");
  note.className = `turn-note ${kind}`;
  note.textContent = text;
  msg.appendChild(note);
}

async function uploadFiles(files) {
  if (!files.length) return [];
  const form = new FormData();
  for (const file of files) form.append("files", file);
  const res = await fetch(`${location.origin}/gradio_api/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return await res.json();
}

function setGenerating(generating) {
  if (generating) {
    sendBtn.textContent = "Stop";
    sendBtn.classList.add("stop");
    // Hide while generating; it reappears only once the turn is in history.
    editLastBtn.hidden = true;
  } else {
    sendBtn.textContent = "Send";
    sendBtn.classList.remove("stop");
    currentJob = null;
  }
}

sendBtn.addEventListener("click", async () => {
  if (currentJob) {
    cancelled = true;
    currentJob.cancel();
    setGenerating(false);
    return;
  }

  const text = messageBox.value.trim();
  const items = pendingFiles;
  if (!text && !items.length) return;
  if (!client) {
    showBanner("Not connected to the server. Please reload the page.");
    return;
  }

  const userEl = addUserMessage(text, items);
  messageBox.value = "";
  messageBox.style.height = "auto";
  // Object URLs stay alive for the rendered bubble; just drop the pending list.
  pendingFiles = [];
  renderPreviews();

  cancelled = false;
  const myEpoch = ++turnEpoch;
  setGenerating(true);
  const els = addAssistantMessage();
  // History length before this turn's entries; kept so the committed turn (below)
  // can be taken back into the composer.
  const historyLenBefore = history.length;

  // Local files are uploaded; remote-URL items (examples) are passed straight
  // to the model. Rebuild the list in the original order.
  let chatFiles;
  try {
    const uploadedPaths = await uploadFiles(items.filter((i) => i.file).map((i) => i.file));
    let next = 0;
    chatFiles = items.map((i) => (i.file ? uploadedPaths[next++] : i.remoteUrl));
  } catch (err) {
    // If the conversation was reset during the upload, it's already cleared
    // (and the reset reset the composer button); don't roll the discarded turn
    // back into the composer or clobber a turn the user has since started.
    if (myEpoch !== turnEpoch) return;
    // Upload failed before the turn even started; roll it back so the user can retry.
    removeTurn(userEl, els.msg);
    loadComposer(text, items);
    showBanner(`Upload failed: ${err.message}`);
    setGenerating(false);
    return;
  }

  // A reset may also have landed during a successful upload; if so, don't start
  // generation for a conversation the user already cleared (the reset already
  // restored the composer button, and a newer turn may now own it).
  if (myEpoch !== turnEpoch) return;

  const job = client.submit("/chat", {
    text,
    files: chatFiles,
    history: history.slice(),
    thinking: thinkingToggle.checked,
    max_new_tokens: Number(maxTokensInput.value),
    image_token_budget: Number(imageBudgetSelect.value),
    system_prompt: systemPromptBox.value,
    temperature: Number(temperatureInput.value),
    top_p: Number(topPInput.value),
    top_k: Number(topKInput.value),
    repetition_penalty: Number(repPenaltyInput.value),
  });
  currentJob = job;

  let lastContent = "";
  // Set on failure (error status or a thrown error). The streamed content is
  // left untouched so any partial answer survives; the detail is surfaced below.
  let errorDetail = null;
  try {
    for await (const msg of job) {
      if (msg.type === "data") {
        const [out] = msg.data;
        const stick = isNearBottom();
        const reasoning = out?.reasoning ?? "";
        const content = out?.content ?? "";
        if (reasoning) {
          els.reasoning.hidden = false;
          renderAssistant(els.reasoningBody, reasoning);
        }
        if (content) {
          lastContent = content;
          renderAssistant(els.bubble, content);
        }
        if (stick) scrollToBottom();
      } else if (msg.type === "unexpected_error" || (msg.type === "status" && (msg.stage ?? msg.status?.stage) === "error")) {
        // Terminal failure (e.g. a raised gr.Error). The client closes the
        // iterator on this event, so the for-await exits on its own.
        errorDetail = msg.message ?? msg.status?.message ?? "Generation failed.";
      }
    }
  } catch (err) {
    errorDetail = err.message ?? "Generation failed.";
  }

  // If the conversation was reset (Clear chat) while this turn was still
  // streaming, the reset already cleared its DOM, history, and composer. Bail
  // out instead of resurrecting the discarded turn as ghost history or a stray
  // "Edit last". A deliberate Stop does not bump the epoch, so that path still
  // falls through below and keeps any partial answer.
  if (myEpoch !== turnEpoch) return;

  const hasContent = Boolean(lastContent);
  const hasReasoning = !els.reasoning.hidden;

  if (hasContent || hasReasoning) {
    // Something worth keeping is on screen: a full/partial answer, or reasoning
    // the user may want to read after stopping mid-thought. Keep it rather than
    // yank the turn away.
    if (hasContent) {
      highlightCode(els.bubble);
      addCopyButtons(els.bubble);
      els.reasoning.open = false; // answer is the focus; fold the reasoning away
    } else {
      // Stopped/failed during reasoning with no answer: drop the empty answer
      // bubble (just typing dots) but leave the reasoning expanded to read.
      els.bubble.remove();
    }
    if (hasReasoning) highlightCode(els.reasoningBody);
    // A stop is the user's own doing, so mark it quietly; a failure (e.g. a
    // ZeroGPU timeout) is not, so call it out with a note and a banner.
    if (cancelled) {
      addTurnNote(els.msg, "⏹ Stopped", "stopped");
    } else if (errorDetail) {
      addTurnNote(els.msg, `⚠ ${errorDetail}`, "error");
      showBanner(errorDetail);
    }
    // Keep the user message so a follow-up has context, but never push an empty
    // assistant turn (it would feed the model a phantom answer). Reasoning is
    // intentionally not stored, matching a normally finished turn.
    history.push({ role: "user", text, files: chatFiles });
    if (hasContent) history.push({ role: "assistant", text: lastContent, files: [] });
    lastTurn = { text, items, userEl, assistantEl: els.msg, historyLenBefore };
    editLastBtn.hidden = false; // turn is recorded; safe to offer take-back
  } else {
    // Nothing was rendered at all (stopped or failed before any output). Roll the
    // turn back into the composer instead of leaving an empty bubble behind.
    removeTurn(userEl, els.msg);
    // Restore the input for a quick retry, unless the user has already started
    // composing something else (don't clobber their draft).
    if (!messageBox.value.trim() && !pendingFiles.length) loadComposer(text, items);
    if (!cancelled && errorDetail) showBanner(errorDetail);
    lastTurn = null;
    editLastBtn.hidden = true;
  }
  setGenerating(false);
});

// Render the empty state (with examples) on first load.
showEmptyState();
