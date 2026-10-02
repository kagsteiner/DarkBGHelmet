# Training-label generator. Runs inside gnubg:
#   BG_SEED=1 BG_OUT=data/labels/w1.txt BG_PLIES=2 BG_COUNT=100000 \
#     gnubg -t -q -p tools/gnubg_label.py
#
# gnubg plays games against itself (best move most of the time, sometimes a
# near-best or random move for diversity). For every roll, `hint` evaluates all
# legal moves; each evaluated resulting position becomes one label line:
#   <posid of position after the move, opponent on roll> <plies> <p0> .. <p4>
# with probabilities (win, win g, win bg, lose g, lose bg) from the view of the
# player now on roll.
import os, random, sys, time
import gnubg

seed = int(os.environ.get("BG_SEED", "1"))
out_path = os.environ.get("BG_OUT", "labels.txt")
plies = int(os.environ.get("BG_PLIES", "2"))
count = int(os.environ.get("BG_COUNT", "1000000"))
accept = int(os.environ.get("BG_ACCEPT", "12"))
rng = random.Random(seed)

gnubg.command("set threads 1")
gnubg.command("new session")
gnubg.command("set evaluation chequerplay evaluation plies %d" % plies)
gnubg.command("set evaluation chequerplay evaluation cubeful off")
if plies >= 1:
    gnubg.command("set evaluation movefilter %d 0 %d 0 0.16" % (plies, accept))
for lvl in range(1, plies):
    gnubg.command("set evaluation movefilter %d %d -1 0 0" % (plies, lvl))

START = gnubg.positionfromid("4HPwATDgc/ABMA")


def parse_pt(t):
    t = t.rstrip("*").lower()
    if t in ("bar", "25"):
        return 24
    if t in ("off", "0"):
        return -1
    return int(t) - 1


def apply_move(board, text):
    """board = (opp, me) in gnubg orientation. Returns (me_after, opp_after)."""
    opp = list(board[0])
    me = list(board[1])
    for tok in text.split():
        rep = 1
        if "(" in tok:
            tok, r = tok.split("(")
            rep = int(r.rstrip(")"))
        pts = [parse_pt(x) for x in tok.split("/")]
        for _ in range(rep):
            for a, b in zip(pts, pts[1:]):
                me[a] -= 1
                if b >= 0:
                    me[b] += 1
                    o = 23 - b
                    if opp[o] == 1:
                        opp[o] = 0
                        opp[24] += 1
    return me, opp


def invert(p):
    return (1.0 - p[0], p[3], p[4], p[1], p[2])


written = 0
t0 = time.time()
with open(out_path, "a", buffering=1) as out:
    board = START
    while written < count:
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        if sum(board[1]) == 15 and sum(board[0]) == 15 and board == START and d1 == d2:
            continue  # no doubles on the opening roll
        gnubg.command("set board " + gnubg.positionid(board))
        gnubg.command("set dice %d%d" % (d1, d2))
        try:
            hint = gnubg.hint(1000)["hint"]
        except Exception:
            hint = []
        if not hint:
            # no legal move: pass the turn
            board = (board[1], board[0])
            continue
        lines = []
        afters = []
        for h in hint:
            me_after, opp_after = apply_move(board, h["move"])
            afters.append((me_after, opp_after))
            if sum(me_after) == 0:
                continue  # game over: no label needed
            pl = h["context"]["plies"] if h.get("type") == "eval" else -1
            pid = gnubg.positionid((me_after, opp_after))
            q = invert(h["details"]["probs"])
            lines.append("%s %d %.6f %.6f %.6f %.6f %.6f\n" % ((pid, pl) + tuple(q)))
        out.write("".join(lines))
        written += len(lines)
        # choose the move to continue the game
        r = rng.random()
        if r < 0.80 or len(hint) == 1:
            k = 0
        elif r < 0.95:
            close = [i for i, h in enumerate(hint[:6]) if h["eqdiff"] > -0.08]
            k = rng.choice(close)
        else:
            k = rng.randrange(len(hint))
        me_after, opp_after = afters[k]
        if sum(me_after) == 0:
            board = START
            if rng.random() < 0.5:
                board = (board[1], board[0])
            continue
        board = (me_after, opp_after)  # opponent (index 1) now on roll

sys.stderr.write("done %d labels in %.0fs\n" % (written, time.time() - t0))
