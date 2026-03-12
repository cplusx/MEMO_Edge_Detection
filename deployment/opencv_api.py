from __future__ import annotations

from typing import Dict

import numpy as np

from deployment.memo_runtime import OptimizedMEMOPredictor


class OpenCVMEMOEdgeDetector:
    def __init__(self, **runtime_kwargs) -> None:
        self.runtime = OptimizedMEMOPredictor(**runtime_kwargs)

    def predict(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        return self.runtime.predict_bgr(image_bgr)

    def predict_file(self, image_path: str) -> Dict[str, np.ndarray]:
        return self.runtime.predict_file(image_path)

    def predict_folder(self, test_folder: str, save_folder: str, batch_size: int = 4, overwrite: bool = False):
        return self.runtime.predict_folder(
            test_folder=test_folder,
            save_folder=save_folder,
            batch_size=batch_size,
            overwrite=overwrite,
        )