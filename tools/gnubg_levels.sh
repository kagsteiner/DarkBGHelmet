#!/bin/sh
# Money-game matches against gnubg at equal depth, in parallel processes.
#   tools/gnubg_levels.sh <our-plies> <gnubg-plies> <games-per-process> [processes] [tag]
# Logs: data/vsgnu/<tag>_<n>.log; summary: tools/agg_matches.py data/vsgnu/<tag>_*.log
cd "$(dirname "$0")/.."
OP=$1; GP=$2; G=$3; N=${4:-16}; TAG=${5:-L${OP}v${GP}}
mkdir -p data/vsgnu
rm -f data/vsgnu/${TAG}_*.log
i=1
while [ $i -le $N ]; do
  BG_GAMES=$G BG_OURPLIES=$OP BG_GNUPLIES=$GP BG_THREADS=1 BG_CUBE_EXTRA=0 BG_SEED=$((10000 * OP + 100 * GP + i)) \
    BG_LOG=data/vsgnu/${TAG}_$i.log gnubg -t -q -p tools/play_vs_gnubg.py > /dev/null 2>&1 &
  i=$((i + 1))
done
wait
echo "== our ${OP}-ply vs gnubg ${GP}-ply" > data/vsgnu/${TAG}_summary.txt
python3 tools/agg_matches.py data/vsgnu/${TAG}_[0-9]*.log >> data/vsgnu/${TAG}_summary.txt
cat data/vsgnu/${TAG}_summary.txt
