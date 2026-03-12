# ONNX Runtime Inference

This directory contains the ONNX Runtime inference workflow for MEMO.

## Menu

- [Notebook Example](#notebook-example)
- [Recommended Runtime](#recommended-runtime)
- [Export ONNX Models](#export-onnx-models)
- [LoRA Finetuned Models](#lora-finetuned-models)
- [Run Folder Inference](#run-folder-inference)
- [First Run Note](#first-run-note)
- [Key Options](#key-options)
- [OpenCV Example](#opencv-example)

## Notebook Example

The quickest way to try the ONNX pipeline is the notebook example:

- [deployment_onnx/onnx_runtime_example.ipynb](deployment_onnx/onnx_runtime_example.ipynb)

The notebook downloads a public example image automatically, runs one image, shows the result inline, and saves outputs in `experiments/onnx_notebook_demo`.

## Recommended Runtime

Generate the runtime recommendation for the current machine:

Run this inside your own prepared Python environment:

```bash
python deployment_onnx/print_runtime_recommendation.py \
  --write_json deployment_onnx/runtime_recommendation.json
```

See [deployment_onnx/runtime_recommendation.json](deployment_onnx/runtime_recommendation.json) for the detected CUDA version and the selected deployment variant.

## Export ONNX Models

Run this inside your own prepared Python environment:

Export the recommended FP16 ONNX models:

```bash
python deployment_onnx/export_onnx.py \
  --config_file configs/binary/discrete_v2data_binary_dinov2.yaml \
  --model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt \
  --output_dir onnx_models/memo_synthetic_late_fp16 \
  --height 352 \
  --width 512 \
  --precision fp16_direct
```

Main output files:

- [onnx_models/memo_synthetic_late_fp16/dino_encoder_fp16.onnx](onnx_models/memo_synthetic_late_fp16/dino_encoder_fp16.onnx)
- [onnx_models/memo_synthetic_late_fp16/memo_denoiser_fp16.onnx](onnx_models/memo_synthetic_late_fp16/memo_denoiser_fp16.onnx)

If you do not want to export locally, you can download pre-exported ONNX models from Hugging Face:

- `cplusx/MEMO_onnx_runtime`

Download one model folder into the local `onnx_models` directory:

```bash
huggingface-cli download cplusx/MEMO_onnx_runtime \
  memo_synthetic_late_fp16/dino_encoder_fp16.onnx \
  memo_synthetic_late_fp16/memo_denoiser_fp16.onnx \
  --repo-type model \
  --local-dir onnx_models
```

Download all exported ONNX folders:

```bash
huggingface-cli download cplusx/MEMO_onnx_runtime \
  --repo-type model \
  --local-dir onnx_models
```

Available ONNX folders in the Hugging Face repository:

- `memo_synthetic_early_fp16`
- `memo_synthetic_late_fp16`
- `memo_bsds_early_lora_fp16`
- `memo_bsds_late_lora_fp16`
- `memo_biped_late_lora_fp16`

## LoRA Finetuned Models

For LoRA finetuned checkpoints, keep using the finetuned checkpoint as `--model_path` and also pass the synthetic pretrained checkpoint as `--base_model_path`.

BSDS example:

```bash
python deployment_onnx/export_onnx.py \
  --config_file configs/discrete_BSDS_finetune/binary_lora_default.yaml \
  --model_path pretrained_models/MEMO_BSDS_ft_late/mp_rank_00_model_states.pt \
  --base_model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt \
  --output_dir onnx_models/memo_bsds_late_lora_fp16 \
  --height 352 \
  --width 512 \
  --precision fp16_direct
```

BIPED example:

```bash
python deployment_onnx/export_onnx.py \
  --config_file configs/discrete_BIPED_finetune/binary_lora_default.yaml \
  --model_path pretrained_models/MEMO_BIPED_ft/mp_rank_00_model_states.pt \
  --base_model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt \
  --output_dir onnx_models/memo_biped_late_lora_fp16 \
  --height 352 \
  --width 512 \
  --precision fp16_direct
```

After export, run ONNX Runtime inference with the exported LoRA ONNX files.

## Run Folder Inference

Run this inside your own prepared Python environment:

```bash
python deployment_onnx/run_onnx_inference.py \
  --test_folder edge_data/BSDS500/BSDS500/data/images/test \
  --save_folder experiments/bsds_test_onnx \
  --dino_encoder_path onnx_models/memo_synthetic_late_fp16/dino_encoder_fp16.onnx \
  --denoiser_path onnx_models/memo_synthetic_late_fp16/memo_denoiser_fp16.onnx \
  --guidance_scale 1.4 \
  --max_steps 20 \
  --resize_long_side 320
```

LoRA example:

```bash
python deployment_onnx/run_onnx_inference.py \
  --test_folder edge_data/BSDS500/BSDS500/data/images/test \
  --save_folder experiments/bsds_lora_test_onnx \
  --dino_encoder_path onnx_models/memo_bsds_late_lora_fp16/dino_encoder_fp16.onnx \
  --denoiser_path onnx_models/memo_bsds_late_lora_fp16/memo_denoiser_fp16.onnx \
  --guidance_scale 1.4 \
  --max_steps 20 \
  --resize_long_side 320
```

For large test images, especially BIPED, `--resize_long_side 320` is strongly recommended.

Output folders:

- [experiments/bsds_test_onnx/prediction](experiments/bsds_test_onnx/prediction)
- [experiments/bsds_test_onnx/binarized](experiments/bsds_test_onnx/binarized)

If the save folder still contains old `quantized` or `colorized` directories from earlier runs, they are moved into `_legacy_outputs/` automatically before writing the new output structure.

## First Run Note

The first ONNX Runtime run can be much slower than later runs.

This is expected. Later runs are usually much faster after the cache is created.

## Key Options

- `--dino_encoder_path`: path to the exported DINO ONNX model
- `--denoiser_path`: path to the exported denoiser ONNX model
- `--base_model_path`: required during export for LoRA finetuned checkpoints
- `--guidance_scale`: classifier-free guidance scale
- `--max_steps`: number of denoising steps
- `--resize_long_side`: resize the input before inference for better speed

Recommended starting values:

- `guidance_scale=1.4`
- `max_steps=20`
- `resize_long_side=320`

## OpenCV Example

```python
import cv2
from deployment_onnx import OpenCVONNXMEMOEdgeDetector

detector = OpenCVONNXMEMOEdgeDetector(
    dino_encoder_path="onnx_models/memo_synthetic_late_fp16/dino_encoder_fp16.onnx",
    denoiser_path="onnx_models/memo_synthetic_late_fp16/memo_denoiser_fp16.onnx",
    guidance_scale=1.4,
    max_steps=20,
    resize_long_side=320,
)

image = cv2.imread("path/to/image.jpg")
result = detector.predict(image)

cv2.imwrite("memo_prediction.png", result["prediction"])
cv2.imwrite("memo_binarized.png", result["binarized"])
```