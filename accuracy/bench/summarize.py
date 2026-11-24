#!/usr/bin/env python3
"""Summarises the timing files written by accuracy/bench.sh (or by running
the two ftbench programs by hand with the same columns).

usage: summarize.py [--markdown | --brief] <bench.tsv>...

Input columns, tab-separated: build (path of the program), tool, model,
corpus, k, threads, texts, bytes, predictions returned, load ms, ns per text,
texts per second, peak memory in MB.

For every model, corpus, k and thread count it prints the time per text of
fastText (its own build flags, "shipped", and the reference flags, "strict")
and of the Go port (plain and GOEXPERIMENT=simd), and how many times faster
the Go port is than fastText as shipped. It also checks that all four
programs returned the same number of predictions.

--brief prints one Markdown table per file with one row per model, for k = 1:
microseconds per text on one thread for both corpora, and how much the
throughput grows with all threads."""
import os
import sys

BUILDS = [
    ("head-shipped", "fastText"),
    ("head-strict", "fastText strict"),
    ("go/", "Go"),
    ("go-simd/", "Go simd"),
]


def build_name(path):
    for key, name in BUILDS:
        if key in path:
            return name
    return path


def us(ns):
    v = ns / 1000
    return f"{v:.1f}" if v < 100 else f"{v:.0f}"


def brief(title, rows, order):
    """One row per model: k = 1, one thread, short texts and paragraphs."""
    corpora = ["tatoeba-golden", "bench-paragraphs"]
    models = []
    for key in order:
        if key[0] not in models:
            models.append(key[0])
    many = max(int(key[3]) for key in order)
    print(f"## {title}\n")
    head = ["Model"]
    for c in ("sentences", "paragraphs"):
        head += [f"fastText, {c}", "Go", "Go simd"]
    head += [f"Go, {many} threads", f"fastText, {many} threads"]
    print("| " + " | ".join(head) + " |")
    print("|---|" + "---:|" * (len(head) - 1))
    for model in models:
        cells = [f"`{model}`"]
        for corpus in corpora:
            r = rows.get((model, corpus, "1", "1"), {})
            base = r.get("fastText")
            cells.append(us(base["ns"]) if base else "")
            for n in ("Go", "Go simd"):
                if n in r and base:
                    cells.append(f"{us(r[n]['ns'])} ({base['ns'] / r[n]['ns']:.1f}×)")
                else:
                    cells.append("")
        one = rows.get((model, corpora[0], "1", "1"), {})
        all_ = rows.get((model, corpora[0], "1", str(many)), {})
        for n in ("Go", "fastText"):
            if n in one and n in all_:
                cells.append(f"{all_[n]['rate'] / one[n]['rate']:.1f}×")
            else:
                cells.append("")
        print("| " + " | ".join(cells) + " |")
    print()


def main():
    flags = ("--markdown", "--brief")
    args = [a for a in sys.argv[1:] if a not in flags]
    markdown = "--markdown" in sys.argv
    for path in args:
        rows = {}
        order = []
        for line in open(path):
            f = line.rstrip("\n").split("\t")
            if len(f) != 13:
                continue
            build = build_name(f[0])
            key = (f[2], f[3].replace(".txt", ""), f[4], f[5])
            if key not in rows:
                rows[key] = {}
                order.append(key)
            rows[key][build] = {
                "returned": f[8],
                "load": float(f[9]),
                "ns": float(f[10]),
                "rate": float(f[11]),
                "rss": float(f[12]),
            }
        names = [name for _, name in BUILDS]
        title = os.path.basename(path).replace(".tsv", "")
        for key in order:
            if len({v["returned"] for v in rows[key].values()}) != 1:
                print(f"{title}: {key}: the programs returned different numbers of predictions", file=sys.stderr)
        if "--brief" in sys.argv:
            brief(title, rows, order)
            continue
        head = ["model", "corpus", "k", "threads"] + [n + " ns" for n in names] + ["Go ×", "Go simd ×", "load ms (fastText / Go)", "MB (fastText / Go)"]
        out = []
        for key in order:
            r = rows[key]
            cells = list(key)
            for n in names:
                cells.append(f"{r[n]['ns']:.0f}" if n in r else "")
            base = r.get("fastText")
            for n in ("Go", "Go simd"):
                cells.append(f"{base['ns'] / r[n]['ns']:.2f}" if base and n in r else "")
            if base and "Go" in r:
                cells.append(f"{base['load']:.0f} / {r['Go']['load']:.0f}")
                cells.append(f"{base['rss']:.0f} / {r['Go']['rss']:.0f}")
            else:
                cells += ["", ""]
            out.append(cells)
        print(f"## {title}\n" if markdown else f"== {title}")
        if markdown:
            print("| " + " | ".join(head) + " |")
            print("|" + "|".join(["---"] * 4 + ["---:"] * (len(head) - 4)) + "|")
            for cells in out:
                print("| " + " | ".join(cells) + " |")
        else:
            widths = [max(len(head[i]), *(len(c[i]) for c in out)) for i in range(len(head))]
            print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
            for cells in out:
                print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        print()


if __name__ == "__main__":
    main()
