#!/usr/bin/env python3
"""
text_overlap_check.py -- self-check of verbatim text reuse before plagiarism screening.

Approximates what iThenticate/Turnitin flag: runs of >= N consecutive identical words.
Reports, for each source, the percentage of the manuscript's words covered by shared
N-grams and the longest shared passages. Numbers-only tokens (table values) are dropped,
so a table of results does not count as copied prose.

    python text_overlap_check.py --manuscript cr_text.txt --sources bhi.txt cited.txt --n 8

This is a proxy, not a substitute for the conference's own similarity report.
"""
import argparse
import re


def tokens(text):
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    toks = re.findall(r"[a-z][a-z0-9\-']*", text.lower())
    return toks


def coverage(m, s, n):
    src = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    hit = [False] * len(m)
    for i in range(len(m) - n + 1):
        if tuple(m[i:i + n]) in src:
            for k in range(i, i + n):
                hit[k] = True
    runs, i = [], 0
    while i < len(m):
        if hit[i]:
            j = i
            while j < len(m) and hit[j]:
                j += 1
            runs.append((j - i, " ".join(m[i:j])))
            i = j
        else:
            i += 1
    return 100.0 * sum(hit) / max(1, len(m)), sorted(runs, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    m = tokens(open(a.manuscript, encoding="utf-8").read())
    print(f"manuscript: {len(m)} word tokens, shingle n={a.n}\n")
    for src in a.sources:
        pct, runs = coverage(m, tokens(open(src, encoding="utf-8").read()), a.n)
        print(f"{src}: {pct:5.1f}% of manuscript words in shared {a.n}-grams; "
              f"{len(runs)} passages, longest {runs[0][0] if runs else 0} words")
        for length, txt in runs[:a.show]:
            print(f"   [{length:>3}] {txt[:150]}{'...' if len(txt) > 150 else ''}")
        print()


if __name__ == "__main__":
    main()
