"""Execute repository notebooks with nbclient."""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
import warnings
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import validator
from nbformat.warnings import MissingIDFieldWarning

ROOT = Path(__file__).resolve().parents[1]

if sys.platform.startswith("win"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def discover_notebooks(paths: list[str]) -> list[Path]:
    if paths:
        notebooks = []
        for raw_path in paths:
            path = Path(raw_path)
            resolved = path if path.is_absolute() else (ROOT / path)
            notebooks.append(resolved.resolve())
        return sorted(notebooks)

    return sorted(
        path.resolve()
        for path in ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )


def execute_notebook(path: Path, timeout: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingIDFieldWarning)
        notebook = nbformat.read(path, as_version=4)
        _, notebook = validator.normalize(notebook)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Notebook paths relative to the repo root.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-cell timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebooks = discover_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found.")
        return 0

    start = time.perf_counter()
    for index, notebook in enumerate(notebooks, start=1):
        relative = notebook.relative_to(ROOT)
        print(f"[{index}/{len(notebooks)}] executing {relative}")
        execute_notebook(notebook, timeout=args.timeout)

    duration = time.perf_counter() - start
    print(f"Executed {len(notebooks)} notebooks in {duration:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
