from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_example_assets import ensure_demo_examples, list_demo_examples
from demo_model_registry import list_model_presets, resolve_model_preset
from deployment.memo_runtime import OptimizedMEMOPredictor


MODEL_NAMES = tuple(list_model_presets().keys())


def _load_demo_example_paths() -> Tuple[str, ...]:
    try:
        return tuple(str(path) for path in ensure_demo_examples())
    except Exception:
        return tuple()


DEMO_EXAMPLE_PATHS = _load_demo_example_paths()
PREDICTOR_CACHE: Dict[Tuple[str, str, str, str, str, bool], OptimizedMEMOPredictor] = {}


def get_predictor(
    model_name: str,
    device: str,
    precision: str,
    compile_model: bool,
) -> OptimizedMEMOPredictor:
    preset = resolve_model_preset(model_name)
    cache_key = (
        model_name,
        str(preset["config_file"]),
        str(preset["model_path"]),
        str(preset["base_model_path"]),
        device,
        precision,
        compile_model,
    )
    predictor = PREDICTOR_CACHE.get(cache_key)
    if predictor is None:
        predictor = OptimizedMEMOPredictor(
            config_file=preset["config_file"],
            model_path=preset["model_path"],
            base_model_path=preset["base_model_path"],
            device=device,
            precision=precision,
            enable_compile=compile_model,
            enable_channels_last=True,
        )
        PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def _normalize_resize_long_side(resize_long_side: int) -> int | None:
    return None if resize_long_side <= 0 else int(resize_long_side)


def _format_model_summary(model_name: str) -> str:
    preset = list_model_presets()[model_name]
    return f"Model: {model_name}\n{preset['description']}"


def run_inference(
    image_rgb: np.ndarray,
    model_name: str,
    guidance_scale: float,
    max_steps: int,
    resize_long_side: int,
    device: str,
    precision: str,
    compile_model: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if image_rgb is None:
        raise gr.Error("Please upload an image.")

    predictor = get_predictor(model_name, device, precision, compile_model)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    started_at = time.perf_counter()
    result = predictor.predict_bgr(
        image_bgr,
        guidance_scale=float(guidance_scale),
        max_steps=int(max_steps),
        resize_long_side=_normalize_resize_long_side(int(resize_long_side)),
    )
    elapsed = time.perf_counter() - started_at

    prediction = result["prediction"]
    binarized = result["binarized"]
    summary = (
        f"{_format_model_summary(model_name)}\n"
        f"Elapsed: {elapsed:.2f}s\n"
        f"Guidance scale: {guidance_scale}\n"
        f"Steps: {max_steps}\n"
        f"Resize long side: {_normalize_resize_long_side(int(resize_long_side))}\n"
        f"Device: {device}\n"
        f"Precision: {precision}\n"
        f"Compile enabled: {compile_model}"
    )
    return prediction, binarized, summary


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MEMO Edge Detection Demo") as demo:
        gr.Markdown(
            "# MEMO Edge Detection Demo\n"
            "Upload one image, choose a MEMO checkpoint, and run the native PyTorch model with adjustable hyperparameters.\n\n"
            "Example images are available below for quick testing, or you can upload your own image."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input Image", type="numpy")
                if DEMO_EXAMPLE_PATHS:
                    gr.Examples(
                        label="Example Images",
                        examples=[[path] for path in DEMO_EXAMPLE_PATHS],
                        inputs=[image_input],
                        cache_examples=False,
                    )
                else:
                    gr.Markdown("Example images could not be downloaded automatically. You can still upload your own image.")
                model_name = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_NAMES),
                    value="Synthetic Late",
                )
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=3.0, value=1.4, step=0.1)
                max_steps = gr.Slider(label="Max Steps", minimum=1, maximum=50, value=20, step=1)
                resize_long_side = gr.Slider(
                    label="Resize Long Side (0 disables resize)",
                    minimum=0,
                    maximum=1024,
                    value=320,
                    step=32,
                )
                device = gr.Dropdown(label="Device", choices=["auto", "cuda", "cpu"], value="auto")
                precision = gr.Dropdown(label="Precision", choices=["fp16", "bf16", "fp32"], value="fp16")
                compile_model = gr.Checkbox(label="Enable torch.compile", value=False)
                run_button = gr.Button("Run MEMO")
            with gr.Column(scale=1):
                summary_output = gr.Textbox(label="Run Summary", lines=10)
                prediction_output = gr.Image(label="Prediction", type="numpy")
                binarized_output = gr.Image(label="Binarized", type="numpy")

        gr.Markdown(
            "## Example Sources\n"
            + "\n".join(
                f"- {name}: {metadata['description']}"
                for name, metadata in list_demo_examples().items()
            )
        )

        run_button.click(
            fn=run_inference,
            inputs=[
                image_input,
                model_name,
                guidance_scale,
                max_steps,
                resize_long_side,
                device,
                precision,
                compile_model,
            ],
            outputs=[prediction_output, binarized_output, summary_output],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()