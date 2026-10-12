"""Layer x top-10 readout matrices for two-hop prompts the model answers correctly.

Input: `data/readout_matrices.csv` - top-10 tokens of both lenses at every layer,
read at the last prompt token.

* GPT-2 XL: WikiText J-lens (`gpt2-xl_jacobian_lens.pt`, 339 WikiText-103 prompts,
  max_seq_len 128), fp32 model with a prepended <|endoftext|>.
* Gemma 4 31B: neuronpedia WikiText J-lens
  (`gemma-4-31b/jlens/Salesforce-wikitext/gemma-4-31B_jacobian_lens.pt`, 861
  prompts), fp32 compute on bf16 weights streamed layer by layer, tokenizer BOS.
  Ranks are pre-softcap.

Rows = examples, columns = lenses. Orange marks the unstated intermediate hop
(solid: the hop itself, tint: other members of its category); blue marks the
answer (solid: the answer in any language, tint: other members of its category).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
N_RANKS = 10
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HOP, ANSWER = "#eb6834", "#2a78d6"  # validated categorical slots 2 and 1 (dataviz palette)
STYLE = {  # kind -> (color, alpha, bold)
    "hop": (HOP, 0.9, True),
    "hop_group": (HOP, 0.28, False),
    "answer": (ANSWER, 0.85, True),
    "answer_group": (ANSWER, 0.28, False),
}
EXAMPLES = {
    "warsaw_polish": dict(
        model="GPT-2 XL",
        title="Warsaw → Poland → Polish",
        layers=[*range(18, 47, 2), 47],
        hop={"Poland", "Poles"},
        hop_group={"Belarus", "Azerbai"},
        answer={"Polish"},
        answer_group={"Arabic", "Bulgarian", "Czech", "English", "Esper", "Finnish", "Gael",
                      "German", "Hebrew", "Hungarian", "Lithuan", "Romanian", "Russian",
                      "Sanskrit", "Serbian", "Slav", "Slov", "Sloven", "Spanish", "Ukrainian",
                      "Welsh"},
        legend=("Poland", "other countries", "Polish", "other languages"),
    ),
    "largest_country_moscow": dict(
        model="Gemma 4 31B",
        title="largest country → Russia → Moscow",
        layers=[*range(22, 59, 2), 59],
        hop={"Russia", "Russian"},
        hop_group={"Afghanistan", "Azerbaijan", "Belarus", "Kazakhstan", "Kyrgyz", "Kyrgyzstan",
                   "Lithuania", "Mongolia", "Romania", "Serbia", "Tajikistan", "Turkmenistan",
                   "Ukraine", "Uzbekistan", "Yugoslavia", "Antarctica"},
        answer={"Moscow", "Mosc", "Mosca", "Moscou", "Mosk", "Moskau", "Моск", "Москва",
                "Москве", "Москвы", "موس"},
        answer_group={"Astana", "Beijing", "Brasília", "Canberra", "Delhi", "Islamabad",
                      "Istanbul", "Jakarta", "Leningrad", "Tehran", "تهران", "北京", "北京市",
                      "東京都", "서울", "নয়াদ", "capital", "首都", "столи"},
        legend=("Russia", "other countries", "Moscow (any language)", "other capitals"),
    ),
    "boot_rome": dict(
        model="Gemma 4 31B",
        title="country shaped like a boot → Italy → Rome",
        layers=[*range(22, 59, 2), 59],
        hop={"Italy", "Италии"},
        hop_group={"England", "France", "Romania", "Spain", "Yugoslavia"},
        answer={"Rome", "ROME", "Roma", "rome", "Рим", "рим", "罗马", "ローマ", "रोम", "روم"},
        answer_group={"Bucharest", "Bucure", "Constantinople", "Istanbul", "Москва", "北京",
                      "首都", "столи", "cities", "city", "নয়াদ", "Roman", "Romans", "romanos",
                      "röm"},
        legend=("Italy", "other countries", "Rome (any language)", "other cities"),
    ),
}
LENSES = ["J-lens", "logit lens"]


def kind_of(token: str, spec: dict) -> str | None:
    word = token.strip()
    for kind in STYLE:
        if word in spec[kind]:
            return kind
    return None


def label(token: str) -> str:
    word = token.strip()
    if not word:
        return "↵" if "\n" in token else "␣"  # newline / space tokens
    return word if word.isprintable() and not 0xE000 <= ord(word[0]) <= 0xF8FF else "·"


def draw(ax: plt.Axes, cells: pd.DataFrame, spec: dict, *, upward: bool = False) -> None:
    """Draw one layer x rank matrix; ``upward`` puts the first layer at the bottom."""
    layers = spec["layers"]
    for row, layer in enumerate(layers):
        for _, cell in cells[cells.layer.eq(layer)].iterrows():
            col = cell["rank"] - 1
            kind = kind_of(cell.token, spec)
            if kind is not None:
                color, alpha, _ = STYLE[kind]
                ax.add_patch(Rectangle((col + 0.04, row + 0.06), 0.92, 0.88, facecolor=color,
                                       alpha=alpha, linewidth=0))
            bold = kind is not None and STYLE[kind][2]
            ax.text(col + 0.5, row + 0.5, label(cell.token)[:12], ha="center", va="center",
                    fontsize=8.6, color=INK if kind else MUTED,
                    fontweight="bold" if bold else "normal")
    ax.axhline(len(layers) - 1, color=MUTED, linewidth=0.9)  # model output below
    ax.set_xlim(0, N_RANKS)
    ax.set_ylim((0, len(layers)) if upward else (len(layers), 0))
    ax.set_xticks([c + 0.5 for c in range(N_RANKS)], [str(c + 1) for c in range(N_RANKS)])
    ax.set_yticks([r + 0.5 for r in range(len(layers))],
                  [str(l) for l in layers[:-1]] + [f"{layers[-1]} (output)"])
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    for x in range(1, N_RANKS):
        ax.axvline(x, color=GRID, linewidth=0.6, zorder=0)


def plot(data: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                         "font.family": ["DejaVu Sans", "Arial Unicode MS"]})
    heights = [len(spec["layers"]) for spec in EXAMPLES.values()]
    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(21, 0.33 * sum(heights) + 2.6 * len(EXAMPLES)),
                             gridspec_kw={"hspace": 0.36, "wspace": 0.06, "height_ratios": heights})
    for row_axes, (example, spec) in zip(axes, EXAMPLES.items(), strict=True):
        frame = data[data.example.eq(example)]
        prompt = frame.prompt.iloc[0].replace("\n", " ")
        for ax, lens in zip(row_axes, LENSES, strict=True):
            draw(ax, frame[frame.lens.eq(lens)], spec)
            ax.set_title(lens, loc="left", fontsize=12, color=MUTED)
            ax.set_xlabel("rank at the last prompt token", color=MUTED, fontsize=9)
        row_axes[0].set_ylabel("layer", color=MUTED)
        row_axes[1].tick_params(labelleft=False)
        top = row_axes[0].get_position().y1
        fig.text(0.125, top + 0.034, f"{spec['model']}: {spec['title']}", fontsize=15,
                 fontweight="bold", color=INK)
        fig.text(0.125, top + 0.019, f"“{prompt}”", fontsize=10.5, color=MUTED,
                 style="italic")
        handles = [Patch(facecolor=STYLE[k][0], alpha=STYLE[k][1], label=text)
                   for k, text in zip(STYLE, spec["legend"], strict=True)]
        fig.legend(handles=handles, loc="lower right", ncol=4, frameon=False, fontsize=10,
                   bbox_to_anchor=(0.9, top + 0.03))
    fig.text(0.125, 0.06, "Top-10 vocabulary tokens per layer at the last prompt token; the "
             "last row is the model's own output, and every model answers correctly. "
             "WikiText J-lenses.", fontsize=9.5, color=MUTED)
    return fig


def plot_single(data: pd.DataFrame, example: str, lens: str) -> plt.Figure:
    """One example, one lens, layers increasing upwards."""
    spec = EXAMPLES[example]
    frame = data[data.example.eq(example) & data.lens.eq(lens)]
    prompt = frame.prompt.iloc[0].replace("\n", " ")
    fig, ax = plt.subplots(figsize=(10.5, 0.33 * len(spec["layers"]) + 1.6))
    draw(ax, frame, spec, upward=True)
    ax.set_ylabel("layer", color=MUTED)
    ax.set_xlabel("rank at the last prompt token", color=MUTED, fontsize=9)
    ax.annotate(f"{spec['model']}, {lens}: {spec['title']}", xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 34), textcoords="offset points", fontsize=14, fontweight="bold",
                color=INK)
    ax.annotate(f"\u201c{prompt}\u201d", xy=(0, 1), xycoords="axes fraction", xytext=(0, 14),
                textcoords="offset points", fontsize=10.5, color=MUTED, style="italic")
    handles = [Patch(facecolor=STYLE[k][0], alpha=STYLE[k][1], label=text)
               for k, text in zip(STYLE, spec["legend"], strict=True)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.045), ncol=4,
              frameon=False, fontsize=9.5)
    return fig


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")


def main() -> None:
    data = pd.read_csv(HERE / "data/readout_matrices.csv", keep_default_na=False)
    save(plot(data), "readout_matrices")
    save(plot_single(data, "largest_country_moscow", "J-lens"), "readout_matrix_gemma_moscow_jlens")


if __name__ == "__main__":
    main()
