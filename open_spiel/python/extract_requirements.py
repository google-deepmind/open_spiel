# /open_spiel/python/extract_requirements.py

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # scripts/ → repo root
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
OUTPUT_DIR = PROJECT_ROOT / "requirements"


def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  with open(PYPROJECT, "rb") as f:
    data = tomllib.load(f)

  extras = data["project"]["optional-dependencies"]
  extras["base"] = data["project"]["dependencies"]
  for name, deps in extras.items():
    outfile = OUTPUT_DIR / f"requirements-{name}.txt"
    outfile.write_text("\n".join(deps) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
