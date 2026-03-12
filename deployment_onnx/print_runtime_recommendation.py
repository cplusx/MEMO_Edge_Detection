from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_onnx.runtime_selector import recommend_runtime, write_runtime_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ONNX deployment recommendation for the current CUDA environment")
    parser.add_argument("--write_json", type=str, default=None)
    args = parser.parse_args()

    recommendation = recommend_runtime()
    print(f"cuda_version: {recommendation.cuda_version}")
    print(f"available_providers: {recommendation.available_providers}")
    print(f"preferred_variant: {recommendation.preferred_variant}")
    print(f"fallback_variant: {recommendation.fallback_variant}")
    print(f"notes: {recommendation.notes}")

    if args.write_json:
        write_runtime_matrix(args.write_json)
        print(f"wrote_json: {args.write_json}")


if __name__ == "__main__":
    main()