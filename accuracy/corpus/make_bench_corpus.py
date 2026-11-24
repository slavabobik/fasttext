#!/usr/bin/env python3
"""Writes corpus/bench-paragraphs.txt for the speed tests: one line per
language, made of that language's first 20 sentences in tatoeba-test.tsv
joined with spaces. Languages with fewer than 20 sentences are left out.

The sentence corpus for the speed tests is tatoeba-golden.txt (short texts);
this one stands for longer texts of a single language."""
import os

here = os.path.dirname(os.path.abspath(__file__))
by_lang = {}
with open(os.path.join(here, "tatoeba-test.tsv"), encoding="utf-8") as f:
    for line in f:
        _, lang, text = line.rstrip("\n").split("\t", 2)
        by_lang.setdefault(lang, [])
        if len(by_lang[lang]) < 20:
            by_lang[lang].append(text)
with open(os.path.join(here, "bench-paragraphs.txt"), "w", encoding="utf-8") as out:
    for lang in sorted(by_lang):
        if len(by_lang[lang]) == 20:
            out.write(" ".join(by_lang[lang]) + "\n")
