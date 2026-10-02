# Money-game match between DarkBGHelmet (via the BGEI protocol) and gnubg, with
# the cube. Runs inside gnubg:
#
#   BG_GAMES=200 BG_OURPLIES=2 BG_GNUPLIES=0 BG_SEED=1 BG_THREADS=2 \
#     gnubg -t -q -p tools/play_vs_gnubg.py
#
# Games are played in duplicate pairs: the same dice sequence is replayed with the
# roles swapped, which removes most of the luck. Prints the running result in
# points per game for DarkBGHelmet (positive = DarkBGHelmet wins).
import math, os, random, subprocess, sys
import gnubg

GAMES = int(os.environ.get("BG_GAMES", "100"))
OUR_PLIES = int(os.environ.get("BG_OURPLIES", "2"))
GNU_PLIES = int(os.environ.get("BG_GNUPLIES", "0"))
SEED = int(os.environ.get("BG_SEED", "1"))
THREADS = os.environ.get("BG_THREADS", "2")
NET = os.environ.get("BG_NET", "nets/latest.bgnn")
LOG = os.environ.get("BG_LOG", "")
ENGINE = os.environ.get("BG_ENGINE", "build/bgei")

gnubg.command("set threads 1")
gnubg.command("new session")
START = gnubg.positionfromid("4HPwATDgc/ABMA")

eng = subprocess.Popen([ENGINE, "--net", NET, "--threads", THREADS], stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, text=True, bufsize=1)


def ask(cmd, until):
    eng.stdin.write(cmd + "\n")
    eng.stdin.flush()
    out = []
    while True:
        line = eng.stdout.readline()
        if not line:
            raise RuntimeError("engine died")
        line = line.strip()
        out.append(line)
        if line.startswith(until) or line.startswith("error"):
            return out


ask("bgei", "bgeiok")


def parse_pt(t):
    t = t.rstrip("*").lower()
    if t in ("bar", "25"):
        return 24
    if t in ("off", "0"):
        return -1
    return int(t) - 1


def apply_move(board, text):
    """board = (opp, me), me to move. Returns new board from the opponent's view."""
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
    return (me, opp)


def result_points(winner_side, loser_side):
    """winner_side/loser_side: 25-int lists in own perspective. 1/2/3 points."""
    if sum(loser_side) < 15:
        return 1
    if any(loser_side[i] for i in range(18, 25)):
        return 3
    return 2


def gnubg_cubeinfo(cube, owner_rel):
    # owner_rel: -1 centered, 0 = player to move owns, 1 = opponent owns (gnubg: fMove = 0)
    return gnubg.cubeinfo(cube, owner_rel, 0, 0, (0, 0), 0, 0)


def gnu_ec(plies):
    return gnubg.evalcontext(1, plies, 1, 1, 0.0)


def gnu_cube(board, cube, owner_rel):
    """(gnubg would double, taker should take) from the view of the player to move."""
    r = gnubg.cfevaluate(board, gnubg_cubeinfo(cube, owner_rel), gnu_ec(GNU_PLIES))
    opt, dt, dp = r[0], r[2], r[3]
    # r[0] is the value of the optimal action; it equals min(DT, DP) iff doubling is right
    doubled = abs(opt - min(dt, dp)) < 1e-6
    return doubled, dt <= dp


def gnu_move(board, d1, d2, cube, owner_rel):
    gnubg.command("set board " + gnubg.positionid(board))
    gnubg.command("set dice %d%d" % (d1, d2))
    gnubg.command("set evaluation chequerplay evaluation plies %d" % GNU_PLIES)
    h = gnubg.hint(1)["hint"]
    return h[0]["move"] if h else None


def xg_owner(owner_rel):
    return {-1: "center", 0: "onroll", 1: "opponent"}[owner_rel]


def send(cmd):
    eng.stdin.write(cmd + "\n")
    eng.stdin.flush()


def our_setup(board, cube, owner_rel):
    send("position gnubg " + gnubg.positionid(board))
    send("match length 0 cube %d owner %s" % (cube, xg_owner(owner_rel)))


def our_move(board, d1, d2, cube, owner_rel):
    our_setup(board, cube, owner_rel)
    out = ask("go move dice %d%d plies %d" % (d1, d2, OUR_PLIES), "bestmove")
    mv = out[-1][len("bestmove "):] if out[-1].startswith("bestmove") else None
    return mv if mv and mv != "(no move)" else None


def our_cube(board, cube, owner_rel):
    our_setup(board, cube, owner_rel)
    out = ask("go cube plies %d" % OUR_PLIES, "bestcube")
    line = out[-1]
    dec = line[len("bestcube "):].split(";")[0]
    take = "take pass" not in line
    doubled = dec.startswith("Double") or dec.startswith("Redouble")
    return doubled, take


def play_game(rng_seed, ours_first):
    """Returns points from DarkBGHelmet's view."""
    rng = random.Random(rng_seed)
    board = START
    ours_to_move = ours_first
    cube = 1
    owner_ours = None  # None = centered, True = DarkBGHelmet owns, False = gnubg owns
    first = True
    while True:
        # owner relative to the player to move
        if owner_ours is None:
            orel = -1
        else:
            orel = 0 if owner_ours == ours_to_move else 1
        # cube action (not before the very first roll)
        if not first and orel != 1:
            if ours_to_move:
                dbl, _ = our_cube(board, cube, orel)
                if dbl:
                    _, take = gnu_cube(board, cube, orel)  # gnubg as taker: evaluate from doubler's view
                    if not take:
                        return cube
                    cube *= 2; owner_ours = False
            else:
                dbl, _ = gnu_cube(board, cube, orel)
                if dbl:
                    _, take = our_cube(board, cube, orel)
                    if not take:
                        return -cube
                    cube *= 2; owner_ours = True
        first = False
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        while board == START and d1 == d2:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        orel = -1 if owner_ours is None else (0 if owner_ours == ours_to_move else 1)
        mv = our_move(board, d1, d2, cube, orel) if ours_to_move else gnu_move(board, d1, d2, cube, orel)
        if mv:
            new = apply_move(board, mv)  # (mover, opponent) -> from opponent's view: (opp=mover, me=opponent)
        else:
            new = (board[1], board[0])
        mover_side = new[0]
        if sum(mover_side) == 0:
            pts = result_points(mover_side, new[1]) * cube
            return pts if ours_to_move else -pts
        board = new
        ours_to_move = not ours_to_move


total = 0.0
sq = 0.0
n = 0
master = random.Random(SEED)
logf = open(LOG, "a") if LOG else None
for g in range(GAMES // 2):
    s = master.randrange(1 << 30)
    a = play_game(s, True)
    b = play_game(s, False)
    pair = (a + b) / 2.0
    total += a + b
    sq += pair * pair
    n += 1
    mean = total / (2 * n)
    se = math.sqrt(max(0.0, sq / n - (total / (2 * n)) ** 2) / n) if n > 1 else 0.0
    msg = "pairs %d games %d  DBGH ppg %+.3f  (se %.3f)  last %+d %+d" % (n, 2 * n, mean, se, a, b)
    sys.stderr.write(msg + "\n")
    if logf:
        logf.write(msg + "\n"); logf.flush()
send("quit")
