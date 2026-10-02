# Run inside gnubg:  gnubg -t -q -p tools/gnubg_movelist.py < positions.txt > out.txt
# Input lines: "<posid> <d1d2>". Output: "<posid> <d1d2> <n>\t<move>|<move>|..."
import sys, gnubg
gnubg.command("new session")
gnubg.command("set evaluation chequerplay evaluation plies 0")
for line in sys.stdin:
    pid, dice = line.split()
    gnubg.command("set board " + pid)
    gnubg.command("set dice %s%s" % (dice[0], dice[1]))
    try:
        h = gnubg.hint(100000)["hint"]
        mv = [x["move"] for x in h]
    except Exception:
        mv = []
    sys.stdout.write("RESULT %s %s %d\t%s\n" % (pid, dice, len(mv), "|".join(mv)))
