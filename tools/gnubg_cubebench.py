# Cube-decision benchmark generator (runs inside gnubg). Reads label lines
# ("<posid> <plies> p0..p4") on stdin, keeps positions where the player on roll
# has 55-92% winning chances, and writes gnubg 2-ply cube analyses for several
# scores:  C2 <posid> <len> <s0> <s1> <cube> <owner> <opt> <dt> <dp> <code>
#   owner: -1 centered, 0 player on roll owns, 1 opponent owns
import os, random, sys
import gnubg

rng = random.Random(int(os.environ.get("BG_SEED", "3")))
count = int(os.environ.get("BG_COUNT", "300"))
plies = int(os.environ.get("BG_PLIES", "2"))
gnubg.command("set threads 1")
ec = gnubg.evalcontext(1, plies, 1, 1, 0.0)
configs = [(0, 0, 0, 1, -1), (0, 0, 0, 2, 0), (7, 0, 0, 1, -1), (7, 4, 2, 1, -1), (7, 2, 4, 1, -1),
           (9, 3, 5, 2, 1), (5, 1, 3, 1, -1), (11, 6, 8, 2, 0), (7, 5, 3, 1, -1)]
lines = [l for l in sys.stdin if l.strip()]
rng.shuffle(lines)
done = 0
for l in lines:
    f = l.split()
    w = float(f[2])
    if not (0.55 <= w <= 0.92):
        continue
    b = gnubg.positionfromid(f[0])
    ln, s0, s1, cube, owner = configs[done % len(configs)]
    ci = gnubg.cubeinfo(cube, owner, 0, ln, (s0, s1), 0, 0)
    r = gnubg.cfevaluate(b, ci, ec)
    sys.stdout.write("C2 %s %d %d %d %d %d %f %f %f %d\n" % (f[0], ln, s0, s1, cube, owner, r[0], r[2], r[3], r[4]))
    sys.stdout.flush()
    done += 1
    if done >= count:
        break
