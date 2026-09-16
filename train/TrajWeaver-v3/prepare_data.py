"""Convert mysplit without teacher calls or GPU work."""

import argparse
import json
from pathlib import Path

from trajweaver_v3.data import prepare

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parents[1] / "data/CToMPersu/mysplit")
    parser.add_argument("--output", type=Path, default=HERE / "data/sft")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
