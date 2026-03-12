# Deployment Runtime

这个目录单独承载推理加速和部署接口，不改动原来的训练代码与 `edge_prediction.py`。

## 设计目标

- 与现有训练/研究代码逻辑分离。
- 在不改模型结构和 checkpoint 格式的前提下提高推理吞吐。
- 提供可直接被 OpenCV 图像流调用的 Python 接口。

## 目前做了什么

- `memo_runtime.py`
  - 按输入分辨率分桶后批量推理，避免 `edge_prediction.py` 当前逐张串行。
  - 预留 `fp16` / `bf16` / `fp32` 三种精度开关。
  - 支持 `channels_last` 和 `torch.compile` 可选加速。
  - 支持 CPU 场景的动态量化 `dynamic-int8`，主要作用在线性层较多的 DINOv2 分支。
  - 支持部署时的 `resize_long_side`，先按长边缩放后推理，再回采样到原图尺寸。
- `opencv_api.py`
  - 提供 `OpenCVMEMOEdgeDetector`，输入直接是 OpenCV 的 `BGR numpy.ndarray`。
- `run_folder_inference.py`
  - 提供面向目录的独立命令行入口。

## 推荐用法

在 GPU 上优先尝试：

请在你自己的 Python 环境中运行下面的命令：

```bash
python deployment/run_folder_inference.py \
  --test_folder edge_data/BSDS500/BSDS500/data/images/test \
  --save_folder experiments/bsds_test_deploy \
  --config_file configs/binary/discrete_v2data_binary_dinov2.yaml \
  --model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt \
  --guidance_scale 1.4 \
  --max_steps 20 \
  --batch_size 4 \
  --resize_long_side 320 \
  --compile
```

如果是 CPU 推理，可以尝试：

```bash
python deployment/run_folder_inference.py \
  --test_folder edge_data/BSDS500/BSDS500/data/images/test \
  --save_folder experiments/bsds_test_cpu_int8 \
  --config_file configs/binary/discrete_v2data_binary_dinov2.yaml \
  --model_path pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt \
  --device cpu \
  --precision fp32 \
  --quantization dynamic-int8
```

## OpenCV 调用示例

```python
import cv2
from deployment import OpenCVMEMOEdgeDetector

detector = OpenCVMEMOEdgeDetector(
    config_file="configs/binary/discrete_v2data_binary_dinov2.yaml",
  model_path="pretrained_models/MEMO_synthetic_late/mp_rank_00_model_states.pt",
    device="cuda",
    guidance_scale=1.4,
    max_steps=20,
    precision="fp16",
  resize_long_side=320,
)

image_bgr = cv2.imread("edge_data/BSDS500/BSDS500/data/images/test/100007.jpg")
result = detector.predict(image_bgr)

cv2.imwrite("memo_prediction.png", result["prediction"])
cv2.imwrite("memo_binarized.png", result["binarized"])
```

## 说明

这里的“OpenCV 调用”指的是输入输出、集成接口和图像管线使用 OpenCV。

当前没有直接做成 OpenCV DNN/纯 ONNX 推理后端，原因是这条模型链路包含：

- DINOv2 主干
- 自定义 UNet + 条件分支
- 多步离散去噪循环
- 运行时局部极大值选择逻辑

这条链路要完整稳定地落到 OpenCV DNN，算子支持和控制流改写成本都比较高，短期内收益不如保留 PyTorch 后端并把 batch/精度/内存布局先优化到位。

如果你更看重吞吐而不是满分辨率细节，优先调 `resize_long_side`，它通常比单纯加 batch 更稳定。