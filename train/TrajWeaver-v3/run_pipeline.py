"""Run the reduced local SFT -> paired utility -> Trigger workflow."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/llama31_8b.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/sft")
    parser.add_argument("--train-scenarios", type=int, default=1000)
    parser.add_argument("--dev-scenarios", type=int, default=200)
    parser.add_argument("--train-prefixes", type=int, default=512)
    parser.add_argument("--dev-prefixes", type=int, default=128)
    parser.add_argument("--planner-data", type=Path, default=None,
                        help="validated planner artifact directory for concern-focused Trigger prefixes")
    parser.add_argument("--planner-prefix-type", choices=("concern", "initial"), default="concern")
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("--run-dir must be new or empty; resume individual stages manually")
    if min(args.train_scenarios, args.dev_scenarios, args.train_prefixes, args.dev_prefixes) < 0:
        parser.error("Sample counts must be nonnegative")
    if not all((args.data_dir / f"{split}.jsonl").exists() for split in ("train", "dev", "test")):
        subprocess.run([sys.executable, str(HERE / "prepare_data.py"), "--output", str(args.data_dir)], check=True)

    def run(script, *options):
        command = [sys.executable, str(HERE / script), "--config", str(args.config.resolve()),
                   *map(str, options)]
        print("Running: " + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    run("train.py", "--stage", "sft", "--data-dir", args.data_dir,
        "--output", args.run_dir / "weaver", "--train-scenarios", args.train_scenarios,
        "--dev-scenarios", args.dev_scenarios)
    best = json.loads((args.run_dir / "weaver/best.json").read_text())["checkpoint"]
    utility_options = ["--checkpoint", best, "--data-dir", args.data_dir,
                       "--output", args.run_dir / "utility", "--train-prefixes", args.train_prefixes,
                       "--dev-prefixes", args.dev_prefixes]
    if args.planner_data:
        utility_options.extend(["--planner-data", args.planner_data,
                                "--planner-prefix-type", args.planner_prefix_type])
    run("build_trigger_data.py", *utility_options)
    run("train.py", "--stage", "trigger", "--checkpoint", best,
        "--data-dir", args.run_dir / "utility", "--output", args.run_dir / "trigger")
    print("Selected final checkpoint: " +
          json.loads((args.run_dir / "trigger/best.json").read_text())["checkpoint"])


if __name__ == "__main__":
    main()
