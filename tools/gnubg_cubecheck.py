# Run inside gnubg with label lines on stdin. For several match/cube settings
# prints gnubg's 0-ply cubeless probabilities and cubeful ND/DT/DP (EMG):
# "C <posid> <len> <s0> <s1> <cube> <owner> <crawford> p0..p4 nd dt dp"
import sys, random, gnubg
rng = random.Random(5)
ec = gnubg.evalcontext(1, 0, 1, 1, 0.0)
configs = [(0,0,0,1,-1,0),(0,0,0,2,0,0),(7,0,0,1,-1,0),(7,5,2,1,-1,0),(7,2,5,1,-1,0),(5,3,3,1,-1,0),
           (9,0,6,1,-1,0),(3,1,1,1,-1,0),(11,8,9,2,0,0),(7,1,4,2,1,0),(5,4,1,1,-1,1),(5,1,4,1,-1,0),(25,10,12,1,-1,0)]
lines = [l for l in sys.stdin if l.strip()]
rng.shuffle(lines)
for l in lines[:400]:
    pid = l.split()[0]
    b = gnubg.positionfromid(pid)
    for (ln, s0, s1, cube, owner, crw) in configs:
        ci = gnubg.cubeinfo(cube, owner, 0, ln, (s0, s1), crw, 0)
        r = gnubg.cfevaluate(b, ci, ec)
        p = gnubg.evaluate(b, ci, gnubg.evalcontext(0, 0, 1, 1, 0.0))
        sys.stdout.write("C %s %d %d %d %d %d %d %f %f %f %f %f %f %f %f\n" % (pid, ln, s0, s1, cube, owner, crw,
                         p[0], p[1], p[2], p[3], p[4], r[0], r[2], r[3]))
