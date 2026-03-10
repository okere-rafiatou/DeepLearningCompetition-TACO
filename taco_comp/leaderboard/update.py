"""
leaderboard/update.py — DO NOT TOUCH

Rebuilds leaderboard/README.md from scores.json.
Called automatically by GitHub Actions.
"""

import json
from datetime import datetime, timezone

SCORES_PATH      = "scores.json"
LEADERBOARD_PATH = "leaderboard/README.md"


def build():
    with open(SCORES_PATH, "r") as f:
        scores = json.load(f)

    entries = sorted(
        scores.values(),
        key=lambda x: x["mAP50"],
        reverse=True,
    )

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Live Leaderboard — TACO Waste Detection",
        "",
        f"*Last updated: {updated}*",
        "",
        "| Rank | Name | mAP50 | mAP50-95 | Submitted at |",
        "|------|------|-------|----------|--------------|",
    ]

    if entries:
        for i, e in enumerate(entries, 1):
            lines.append(
                f"| {i} | {e['name']} | {e['mAP50']:.4f} "
                f"| {e['mAP50-95']:.4f} | {e['submitted_at']} |"
            )
    else:
        lines.append("| — | — | — | — | — |")

    lines += [
        "",
        "---",
        "",
        "*Rankings update automatically after each valid submission.*",
    ]

    with open(LEADERBOARD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Leaderboard updated: {len(entries)} submission(s).")


if __name__ == "__main__":
    build()
