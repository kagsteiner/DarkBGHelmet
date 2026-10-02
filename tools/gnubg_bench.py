# Checker-play benchmark generator. Runs inside gnubg:
#   BG_SEED=1 BG_OUT=data/bench/b1.txt BG_COUNT=1000 gnubg -t -q -p tools/gnubg_bench.py
#
# gnubg plays money games against itself (with some randomness). For each
# non-trivial roll (at least 2 legal plays) it evaluates ALL plays at gnubg
# 2-ply (= XG 3-ply), cubeful, centered cube, and writes one line:
#   <posid> <d1><d2> <n> <move1>=<equity1>|<move2>=<equity2>|...
# plus the move gnubg 0-ply would choose (reference for "how good is 0-ply"):
#   ... \t<best0>
import os, random, sys
import gnubg

seed = int(os.environ.get("BG_SEED", "1"))
out_path = os.environ.get("BG_OUT", "bench.txt")
count = int(os.environ.get("BG_COUNT", "1000"))
rng = random.Random(seed)
gnubg.command("set threads 1")
gnubg.command("new session")
gnubg.command("set evaluation chequerplay evaluation cubeful on")
START = gnubg.positionfromid("4HPwATDgc/ABMA")


def parse_pt(t):
    t = t.rstrip("*").lower()
    if t in ("bar", "25"):
        return 24
    if t in ("off", "0"):
        return -1
    return int(t) - 1


def apply_move(board, text):
    opp = list(board[0]); me = list(board[1])
    for tok in text.split():
        rep = 1
        if "(" in tok:
            tok, r = tok.split("("); rep = int(r.rstrip(")"))
        pts = [parse_pt(x) for x in tok.split("/")]
        for _ in range(rep):
            for a, b in zip(pts, pts[1:]):
                me[a] -= 1
                if b >= 0:
                    me[b] += 1
                    o = 23 - b
                    if opp[o] == 1:
                        opp[o] = 0; opp[24] += 1
    return me, opp


def hint(plies, accept):
    gnubg.command("set evaluation chequerplay evaluation plies %d" % plies)
    if plies >= 1:
        gnubg.command("set evaluation movefilter %d 0 %d 0 0.0" % (plies, accept))
        for lvl in range(1, plies):
            gnubg.command("set evaluation movefilter %d %d -1 0 0" % (plies, lvl))
    return gnubg.hint(1000)["hint"]


written = 0
with open(out_path, "a", buffering=1) as out:
    board = START
    while written < count:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        if board == START and d1 == d2:
            continue
        pid = gnubg.positionid(board)
        gnubg.command("set board " + pid)
        gnubg.command("set dice %d%d" % (d1, d2))
        try:
            h2 = hint(2, 1000)
        except Exception:
            h2 = []
        if not h2:
            board = (board[1], board[0])
            continue
        if len(h2) >= 2 and all(x["context"]["plies"] == 2 for x in h2):
            # gnubg caches the hint per position: set the board again before the 0-ply hint
            gnubg.command("set board " + pid)
            gnubg.command("set dice %d%d" % (d1, d2))
            h0 = hint(0, 0)
            out.write("%s %d%d %d %s\t%s\n" % (pid, d1, d2, len(h2),
                      "|".join("%s=%.5f" % (x["move"], x["equity"]) for x in h2), h0[0]["move"]))
            written += 1
        moves = h2
        r = rng.random()
        k = 0 if (r < 0.85 or len(moves) == 1) else rng.randrange(min(4, len(moves)))
        me_after, opp_after = apply_move(board, moves[k]["move"])
        if sum(me_after) == 0:
            board = START if rng.random() < 0.5 else START
            continue
        board = (me_after, opp_after)
