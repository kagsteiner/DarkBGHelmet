#!/usr/bin/env python3
"""Combine the logs of parallel play_vs_gnubg.py runs into one result.

    tools/agg_matches.py data/vsgnu/L3_*.log
"""
import math, re, sys

pairs = 0
wsum = 0.0
var = 0.0
stats = {}
times = {}
for path in sys.argv[1:]:
    last = None
    timing = None
    for line in open(path):
        if line.startswith("pairs "):
            last = line
        elif line.startswith("TIMING "):
            timing = line
    if not last:
        continue
    m = re.match(r"pairs (\d+) games \d+\s+DBGH ppg ([-+0-9.]+)\s+\(se ([0-9.]+)\).*?(\{.*\})", last)
    n, ppg, se = int(m.group(1)), float(m.group(2)), float(m.group(3))
    pairs += n
    wsum += n * ppg
    var += (n * se) ** 2
    for k, v in eval(m.group(4)).items():
        stats[k] = stats.get(k, 0) + v
    if timing:
        for who, kind, ms, cnt in re.findall(r"(ours|gnu) (move|cube) ([0-9.]+) ms(?: \([0-9.]+ w/o API overhead\))? n=(\d+)", timing):
            t = times.setdefault((who, kind), [0.0, 0])
            t[0] += float(ms) * int(cnt)
            t[1] += int(cnt)
        ov = re.search(r"overhead ([0-9.]+) ms/call", timing)
        overhead = float(ov.group(1)) if ov else 0.0
if pairs == 0:
    sys.exit("no results")
mean = wsum / pairs
se = math.sqrt(var) / pairs
print("games %d  DarkBGHelmet ppg %+.4f  (se %.4f, %.1f sigma)" % (2 * pairs, mean, se, mean / se if se else 0))
print("cube: %s" % stats)
for who in ("ours", "gnu"):
    for kind in ("move", "cube"):
        t = times.get((who, kind))
        if t and t[1]:
            avg = t[0] / t[1]
            extra = "" if who == "ours" else "  (%.1f ms without gnubg API overhead)" % max(0.0, avg - overhead)
            print("time %-4s %-4s %7.1f ms%s  n=%d" % (who, kind, avg, extra, t[1]))
