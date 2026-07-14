"""Tunable generation parameters shared by the chat API and the UI.

This module is the single source of truth for the generation parameters that
the frontend exposes as sliders/selects. It deliberately avoids heavy imports
(torch, transformers, gradio) so it stays cheap to import from tests and, later,
can be used to drive the frontend controls without pulling in the model.

The chat endpoint is a public API: clients can call it directly, bypassing the
UI controls, so the ranges here are enforced server-side rather than trusted
from the request.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    """Allowed range, default, and UI presentation for one generation parameter.

    The numeric range is what the chat API enforces; ``step`` and ``choices``
    describe how the frontend renders the control. ``validate_params`` only uses
    ``minimum``/``maximum``/``is_int``; the rest is consumed by ``ui_config``.

    Args:
        minimum: Smallest accepted value (inclusive).
        maximum: Largest accepted value (inclusive).
        default: Value used when the client omits the parameter.
        is_int: Whether the value is coerced to ``int`` (otherwise ``float``).
        step: Slider step for the UI control, or ``None`` for a select.
        choices: Discrete options shown in the UI as a select. The API still
            accepts any value within ``[minimum, maximum]``; these are a UI
            convenience, not an extra constraint.
    """

    minimum: float
    maximum: float
    default: float
    is_int: bool = False
    step: float | None = None
    choices: tuple[int, ...] | None = None


# Single source of truth for the tunable parameters. Both the chat API
# (validate_params) and the UI controls (ui_config, injected into the page)
# derive from this, so the ranges/defaults are defined in exactly one place.
PARAM_SPECS: dict[str, ParamSpec] = {
    "max_new_tokens": ParamSpec(minimum=100, maximum=4000, default=2000, is_int=True, step=10),
    "image_token_budget": ParamSpec(
        minimum=70, maximum=1120, default=280, is_int=True, choices=(70, 140, 280, 560, 1120)
    ),
    "temperature": ParamSpec(minimum=0.0, maximum=2.0, default=1.0, step=0.1),
    "top_p": ParamSpec(minimum=0.0, maximum=1.0, default=0.95, step=0.05),
    "top_k": ParamSpec(minimum=0, maximum=100, default=64, is_int=True, step=1),
    "repetition_penalty": ParamSpec(minimum=1.0, maximum=2.0, default=1.0, step=0.05),
}


def ui_config() -> dict[str, dict]:
    """Serialize the param specs for the frontend controls.

    Returns:
        A JSON-serializable dict, keyed by parameter name, describing each
        control's range, step, default, and discrete choices. The page injects
        this as ``window.PARAM_CONFIG`` so the UI and the API share one
        definition of the parameters.
    """
    return {
        name: {
            "min": spec.minimum,
            "max": spec.maximum,
            "step": spec.step,
            "default": spec.default,
            "choices": list(spec.choices) if spec.choices is not None else None,
        }
        for name, spec in PARAM_SPECS.items()
    }


# The page applies window.PARAM_CONFIG to its controls; this tag marks where the
# config script is inserted (just before the frontend module loads).
_APP_SCRIPT_TAG = '<script type="module" src="./app.js"></script>'


def inject_param_config(html: str) -> str:
    """Insert ``window.PARAM_CONFIG`` into the page before the app script.

    Lets the served page and the UI-preview stub share one definition of the
    parameters with the chat API. The injected content is fully controlled
    (numbers and parameter names), so no extra escaping is needed.

    Args:
        html: The page source containing the app script tag.

    Returns:
        The page with a ``<script>`` defining ``window.PARAM_CONFIG`` inserted
        just before the app script.
    """
    blob = f"<script>window.PARAM_CONFIG = {json.dumps(ui_config())};</script>\n    "
    return html.replace(_APP_SCRIPT_TAG, blob + _APP_SCRIPT_TAG)


def validate_params(values: dict[str, float]) -> dict[str, float]:
    """Coerce and range-check generation parameters against ``PARAM_SPECS``.

    Args:
        values: Raw parameter values keyed by name (one per entry in
            ``PARAM_SPECS``).

    Returns:
        A new dict with each value coerced to its declared numeric type.

    Raises:
        ValueError: If a value is non-numeric or outside its allowed range. The
            message names the parameter and its range so the API is
            self-documenting.
    """
    validated: dict[str, float] = {}
    for name, spec in PARAM_SPECS.items():
        raw = values[name]
        try:
            coerced = int(raw) if spec.is_int else float(raw)
        except (TypeError, ValueError):
            expected = "an integer" if spec.is_int else "a number"
            msg = f"{name} must be {expected}, got {raw!r}"
            raise ValueError(msg) from None
        if not spec.minimum <= coerced <= spec.maximum:
            lo = int(spec.minimum) if spec.is_int else spec.minimum
            hi = int(spec.maximum) if spec.is_int else spec.maximum
            msg = f"{name} must be between {lo} and {hi}, got {coerced}"
            raise ValueError(msg)
        validated[name] = coerced
    return validated
