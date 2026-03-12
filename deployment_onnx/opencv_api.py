from __future__ import annotations

from typing import Dict

import numpy as np

from deployment_onnx.onnx_runtime import ONNXMEMOPredictor


class OpenCVONNXMEMOEdgeDetector:
    def __init__(self, **runtime_kwargs) -> None:
        self.runtime = ONNXMEMOPredictor(**runtime_kwargs)

    def predict(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        return self.runtime.predict_bgr(image_bgr)

    def predict_folder(self, test_folder: str, save_folder: str, batch_size: int = 1, overwrite: bool = False):
        return self.runtime.predict_folder(test_folder, save_folder, batch_size=batch_size, overwrite=overwrite)