#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path


def has_episode_json_files(directory: Path):
    return any(directory.glob("episode_*.json"))


def iter_episode_json_files(source_dir: Path):
    if has_episode_json_files(source_dir):
        for path in source_dir.glob("episode_*.json"):
            if path.is_file():
                yield path, path.relative_to(source_dir.parent)
        return

    child_dirs = [path for path in source_dir.iterdir() if path.is_dir()]

    if any(has_episode_json_files(child) for child in child_dirs):
        relative_base = source_dir.parent
        for child_dir in child_dirs:
            for path in child_dir.glob("episode_*.json"):
                if path.is_file():
                    yield path, path.relative_to(relative_base)
        return

    relative_base = source_dir
    for experiment_dir in child_dirs:
        if not experiment_dir.is_dir():
            continue
        for eval_set_dir in experiment_dir.iterdir():
            if not eval_set_dir.is_dir():
                continue
            for path in eval_set_dir.glob("episode_*.json"):
                if path.is_file():
                    yield path, path.relative_to(relative_base)


def copy_episode_json_tree(source_dir: Path, output_dir: Path, dry_run: bool = False):
    copied = 0
    for source_path, relative_path in iter_episode_json_files(source_dir):
        output_path = output_dir / relative_path

        if dry_run:
            print(f"{source_path} -> {output_path}")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy episode JSON files into output while "
            "preserving the original directory tree."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("running/eb_alfred/gpt-5"),
        help="Source benchmark, method, or eval-set directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_folder/eb_alfred"),
        help="Output directory to create/populate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be copied without writing anything.",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    copied = copy_episode_json_tree(source_dir, output_dir, dry_run=args.dry_run)
    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {copied} files from {source_dir} to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
