#!/usr/bin/env python3
"""Summarises compare_builds.sh output.

usage: summarize.py <results.tsv>...

Prints one row per model and build pair, added up over the corpora:

  records      records compared
  ids          records whose ids differ
  hidden       records whose hidden vector differs in any bit
  top1         records whose best label differs (k=1)
  order5       records whose five best labels differ in label or order
  order-all    records whose full ranking differs
  count        records where the number of predictions differs (any k)
  prob1        records whose best probability differs in any bit
  max diff     largest probability difference for the same label (any k)
  flip gap     for records whose best label differs: the largest gap between
               the two best probabilities in the baseline
  ties         records whose two best probabilities are bit-equal in the
               baseline (k=5)
"""

import collections
import sys

COLUMNS = [
    "model", "corpus", "base", "other", "k", "records", "header", "ids",
    "hidden", "hidden_ulp", "count", "top1", "order", "score", "score_ulp",
    "prob", "prob_ulp", "max_prob", "flip_gap", "ties",
]


def main(paths):
    rows = collections.OrderedDict()
    for path in paths:
        with open(path) as f:
            for line in f:
                r = dict(zip(COLUMNS, line.rstrip("\n").split("\t")))
                key = (r["model"], r["base"], r["other"])
                agg = rows.setdefault(key, collections.Counter())
                if r["header"]:
                    agg["header"] += 1
                if r["k"] == "k1":
                    agg["records"] += int(r["records"])
                    agg["ids"] += int(r["ids"])
                    agg["hidden"] += int(r["hidden"])
                    agg["top1"] += int(r["top1"])
                    agg["prob1"] += int(r["prob"])
                if r["k"] == "k5":
                    agg["order5"] += int(r["order"])
                    agg["ties"] += int(r["ties"])
                if r["k"] == "k-1":
                    agg["order_all"] += int(r["order"])
                agg["count"] += int(r["count"])
                agg["max_prob"] = max(agg["max_prob"], float(r["max_prob"]))
                agg["flip_gap"] = max(agg["flip_gap"], float(r["flip_gap"]))

    head = ["model", "baseline", "other", "records", "ids", "hidden", "top1",
            "order5", "order-all", "count", "prob1", "max diff", "flip gap", "ties"]
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for (model, base, other), a in sorted(rows.items()):
        cells = [
            model, base, other, a["records"], a["ids"], a["hidden"], a["top1"],
            a["order5"], a["order_all"], a["count"], a["prob1"],
            "%.2g" % a["max_prob"] if a["max_prob"] else "0",
            "%.2g" % a["flip_gap"] if a["flip_gap"] else "0", a["ties"],
        ]
        if a["header"]:
            cells[0] += " (header differs)"
        print("| " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
