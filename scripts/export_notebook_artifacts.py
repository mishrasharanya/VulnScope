"""Export rendered notebook figures and tables as standalone files."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
from pathlib import Path

import pandas as pd


def slugify(value: str) -> str:
    value = re.sub(r"[`*_#]", "", value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:70] or "output"


def cell_label(cells: list[dict], index: int) -> str:
    for prior in range(index - 1, -1, -1):
        cell = cells[prior]
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", []):
            if line.lstrip().startswith("#"):
                return slugify(line)
    source = "".join(cells[index].get("source", [])).splitlines()
    return slugify(source[0] if source else f"cell-{index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    figures_dir = args.output_dir / "graphs"
    tables_dir = args.output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str | int]] = []
    figure_number = 0
    table_number = 0

    for cell_index, cell in enumerate(notebook["cells"]):
        label = cell_label(notebook["cells"], cell_index)
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})

            if "image/png" in data:
                figure_number += 1
                filename = f"{figure_number:02d}_cell-{cell_index}_{label}.png"
                path = figures_dir / filename
                encoded = data["image/png"]
                if isinstance(encoded, list):
                    encoded = "".join(encoded)
                path.write_bytes(base64.b64decode(encoded))
                manifest.append(
                    {
                        "type": "graph",
                        "notebook_cell": cell_index,
                        "output": output_index,
                        "description": label.replace("-", " "),
                        "file": str(path.relative_to(args.output_dir)),
                    }
                )

            if "text/html" in data:
                html = data["text/html"]
                if isinstance(html, list):
                    html = "".join(html)
                try:
                    frames = pd.read_html(io.StringIO(html))
                except ValueError:
                    frames = []
                for frame in frames:
                    table_number += 1
                    filename = f"{table_number:02d}_cell-{cell_index}_{label}.csv"
                    path = tables_dir / filename
                    frame.to_csv(path, index=False)
                    manifest.append(
                        {
                            "type": "table",
                            "notebook_cell": cell_index,
                            "output": output_index,
                            "description": label.replace("-", " "),
                            "file": str(path.relative_to(args.output_dir)),
                        }
                    )

    manifest_path = args.output_dir / "index.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["type", "notebook_cell", "output", "description", "file"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print(
        f"Exported {figure_number} graphs and {table_number} tables "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
