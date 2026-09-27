"""Check project-owned C/C++ files using the configured compilation database."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--format", default="clang-format-22")
    parser.add_argument("--tidy", default="clang-tidy-22")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    sources = sorted(
        path.resolve()
        for directory in ("include", "src", "tests")
        for path in (root / directory).rglob("*")
        if path.suffix in (".c", ".cpp", ".h", ".hpp")
    )
    # The generator owns this table's layout; generate_constants.py --check
    # verifies it separately instead of reformatting generated output.
    generated = root / "src/algorithms/constants.hpp"
    subprocess.run(
        [args.format, "--dry-run", "--Werror", *map(str, (p for p in sources if p != generated))],
        cwd=root, check=True, timeout=60,
    )
    database = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
    compiled = set()
    for entry in database:
        path = Path(entry["file"])
        if not path.is_absolute():
            path = Path(entry["directory"]) / path
        compiled.add(path.resolve())
    units = [path for path in sources if path.suffix in (".c", ".cpp")]
    missing = set(units) - compiled
    if missing:
        raise SystemExit("Missing compilation commands: " + ", ".join(map(str, sorted(missing))))
    for path in units:
        print(f"clang-tidy: {path.relative_to(root)}", flush=True)
        subprocess.run(
            [args.tidy, f"-p={build}", "--warnings-as-errors=*", str(path)],
            cwd=root, check=True, timeout=180,
        )


if __name__ == "__main__":
    main()
