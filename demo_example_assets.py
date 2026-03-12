from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_EXAMPLE_DIR = PROJECT_ROOT / "experiments" / "demo_examples"


EXAMPLE_IMAGES: Dict[str, Dict[str, str]] = {
    "Lena": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        "filename": "lena.jpg",
        "description": "Classic portrait image from the OpenCV sample set.",
    },
    "Fruits": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
        "filename": "fruits.jpg",
        "description": "Fruit still-life image from the OpenCV sample set.",
    },
    "Smarties": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png",
        "filename": "smarties.png",
        "description": "Colorful candy image from the OpenCV sample set.",
    },
}


def list_demo_examples() -> Dict[str, Dict[str, str]]:
    return EXAMPLE_IMAGES


def ensure_demo_example(name: str) -> Path:
    if name not in EXAMPLE_IMAGES:
        raise KeyError(f"Unknown demo example: {name}")

    metadata = EXAMPLE_IMAGES[name]
    DEMO_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DEMO_EXAMPLE_DIR / metadata["filename"]
    if target_path.exists():
        return target_path

    with urlopen(metadata["url"], timeout=30) as response, open(target_path, "wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return target_path


def ensure_demo_examples() -> List[Path]:
    output_paths: List[Path] = []
    for name in EXAMPLE_IMAGES:
        output_paths.append(ensure_demo_example(name))
    return output_paths