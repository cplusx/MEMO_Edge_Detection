from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checkpoint_registry import get_checkpoint_metadata, list_checkpoint_names, list_checkpoints


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"skip: {destination} already exists")
        return

    tmp_destination = destination.with_suffix(destination.suffix + ".part")
    if tmp_destination.exists():
        tmp_destination.unlink()

    with urlopen(url, timeout=60) as response, open(tmp_destination, "wb") as output_file:
        total_size = response.headers.get("Content-Length")
        total_size_int = int(total_size) if total_size is not None else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)
            if total_size_int:
                percent = downloaded * 100.0 / total_size_int
                print(
                    f"downloading: {destination.name} {percent:6.2f}% "
                    f"({format_size(downloaded)} / {format_size(total_size_int)})",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"downloading: {destination.name} {format_size(downloaded)}",
                    end="\r",
                    flush=True,
                )

    print(" " * 120, end="\r")
    tmp_destination.replace(destination)
    print(f"saved: {destination}")


def resolve_destination(metadata: dict, output_root: Path) -> Path:
    return output_root / str(metadata["folder_name"]) / str(metadata["filename"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MEMO checkpoints from Hugging Face")
    parser.add_argument("--model", dest="models", action="append", choices=list_checkpoint_names(), help="Checkpoint name to download. Repeat to download multiple models.")
    parser.add_argument("--all", action="store_true", help="Download all registered checkpoints.")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints and exit.")
    parser.add_argument("--print-path", action="store_true", help="Print the resolved local checkpoint path for the selected model(s) and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload checkpoints even if the target file already exists.")
    parser.add_argument("--output-root", type=str, default="pretrained_models", help="Root directory where checkpoint folders will be created. Default: pretrained_models")
    return parser


def resolve_output_root(output_root: str) -> Path:
    candidate = Path(output_root).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def main() -> None:
    args = build_parser().parse_args()
    output_root = resolve_output_root(args.output_root)

    if args.list:
        for name, metadata in list_checkpoints().items():
            destination = resolve_destination(metadata, output_root)
            print(f"{name}: {metadata['display_name']}")
            print(f"  folder: {destination.parent}")
            print(f"  url: {metadata['url']}")
            print(f"  description: {metadata['description']}")
        return

    if args.all:
        model_names = list_checkpoint_names()
    else:
        model_names = args.models or []

    if not model_names:
        raise SystemExit("Specify at least one --model name, or use --all, or use --list.")

    if args.print_path:
        for model_name in model_names:
            metadata = get_checkpoint_metadata(model_name)
            destination = resolve_destination(metadata, output_root)
            print(destination)
        return

    for model_name in model_names:
        metadata = get_checkpoint_metadata(model_name)
        destination = resolve_destination(metadata, output_root)
        print(f"start: {model_name} -> {destination}")
        download_file(str(metadata['url']), destination, overwrite=args.overwrite)


if __name__ == "__main__":
    main()