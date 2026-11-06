"""Draw the README chart from results/*.csv (values transcribed from the saved notebook outputs)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ec = pd.read_csv("results/emotion_classification.csv")
lm = pd.read_csv("results/language_models.csv")
C = {"zero": "#9ca3af", "ft": "#1d4ed8", "noise": "#c2410c", "llm": "#2a7f62"}

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150, gridspec_kw={"width_ratios": [1.5, 1, 1]})

a = ax[0]
rows = [("DistilBERT\nzero-shot", ec.iloc[0], C["zero"]), ("DistilRoBERTa\nzero-shot", ec.iloc[1], C["zero"]),
        ("DistilBERT\nfine-tuned", ec.iloc[2], C["ft"]), ("DistilRoBERTa\nfine-tuned", ec.iloc[3], C["ft"]),
        ("Zephyr-7B\nstructured*", ec.iloc[9], C["llm"]), ("GPT-4o-mini\nstructured*", ec.iloc[7], C["llm"])]
for i, (lab, r, col) in enumerate(rows):
    a.barh(i, r.macro_f1, color=col, height=.62)
    a.text(r.macro_f1 + .01, i, f"F1 {r.macro_f1:.3f} | acc {r.accuracy:.3f}", va="center", fontsize=8)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=8); a.invert_yaxis()
a.set_xlim(0, 1.15); a.set_xlabel("macro F1\n*100-tweet subset, not directly comparable with the 1,011-tweet rows", fontsize=8); a.set_title("Tweet emotion (4 classes): macro F1", fontsize=10)

b = ax[1]
for i, (m, c, n) in enumerate([("DistilBERT", ec.iloc[2], ec.iloc[4]), ("DistilRoBERTa", ec.iloc[3], ec.iloc[5])]):
    b.bar(i - .18, c.accuracy, .34, color=C["ft"], label="clean" if i == 0 else None)
    b.bar(i + .18, n.accuracy, .34, color=C["noise"], label="10% word typos" if i == 0 else None)
    b.text(i - .18, c.accuracy + .005, f"{c.accuracy:.3f}", ha="center", fontsize=8)
    b.text(i + .18, n.accuracy + .005, f"{n.accuracy:.3f}", ha="center", fontsize=8)
b.set_xticks([0, 1]); b.set_xticklabels(["DistilBERT", "DistilRoBERTa"]); b.set_ylim(0, 1.0)
b.set_title("Typo stress test (accuracy, axis from 0)", fontsize=10); b.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)

c = ax[2]
vals = [lm.test_ppl[0], lm.test_ppl[1]]
c.bar(["Count trigram\n(add-k)", "Neural trigram\n(MLP)"], vals, color=[C["zero"], C["ft"]], width=.55)
c.set_yscale("log")
for i, v in enumerate(vals): c.text(i, v * 1.1, f"{v:,.0f}", ha="center", fontsize=9)
c.set_ylim(100, 20000); c.set_title("Test perplexity (log scale, lower is better)\nFDR addresses", fontsize=10)

for x in ax: [x.spines[k].set_visible(False) for k in ("top", "right")]
plt.tight_layout(); plt.savefig("assets/results_overview.png")
