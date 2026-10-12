"""Extract the exact pass@10 bootstrap table (cell 27) from each executed notebook.

These saved tables are the only exact numbers for runs whose interval CSVs are
not available locally; they are used to check the digitized curves at k = 10.
"""

import json
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "gpt2-xl": "notebooks/gpt2-xl/latent_bridge_v2/latent_bridge_v2_lens_benchmark_bootstrap.ipynb",
    "gemma-4-31b": "notebooks/gemma-4-31b/gemma31b_latent_bridge_v2_lens_benchmark_bootstrap.ipynb",
    "gemma-4-e4b": "notebooks/gemma-4-e4b/latent_bridge_v2_lens_benchmark_bootstrap.ipynb",
}


def first_html_table(cell: dict) -> pd.DataFrame:
    """Parse the first HTML DataFrame output of a cell."""
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            table = BeautifulSoup("".join(html), "html.parser").find("table")
            rows = [[c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                    for tr in table.find_all("tr")]
            header, body = rows[0], rows[1:]
            frame = pd.DataFrame(body, columns=header)
            return frame.drop(columns=[header[0]])  # drop the index column
    raise ValueError("cell has no HTML table output")


def main() -> None:
    frames = []
    for model, path in NOTEBOOKS.items():
        cells = json.loads((ROOT / path).read_text())["cells"]
        table = first_html_table(cells[27])
        for column in ("estimate", "ci_low", "ci_high"):
            table[column] = pd.to_numeric(table[column], errors="coerce")
        frames.append(table.assign(model=model, k=10))
    exact = pd.concat(frames, ignore_index=True)
    out = ROOT / "poster/data/exact_pass_at_10.csv"
    exact.to_csv(out, index=False)
    print(exact.groupby("model").size().to_string())
    print(pd.read_csv(StringIO(exact.head(3).to_csv(index=False))).to_string())
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
