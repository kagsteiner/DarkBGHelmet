#!/bin/sh
# Expert iteration: relabel a third of the position pool with our own 3-ply search
# using the current best net, retrain on all own labels, duel against the best
# net, and adopt the new net if it wins.
#   tools/iterate.sh <start-net> [first-part] [last-part]
cd "$(dirname "$0")/.."
BEST=${1:-nets/full41m.bgnn}
FIRST=${2:-0}; LAST=${3:-2}
LOG=data/iter.log
TH=17
echo "$(date '+%F %T') start, best=$BEST" >> $LOG
k=$FIRST
while [ $k -le $LAST ]; do
  n=$((k + 1))
  echo "$(date '+%F %T') iteration $n: relabel data/pool/part$k with $BEST" >> $LOG
  ./build/bgtrain relabel data/own/own$n.txt --net $BEST --plies 3 --threads $TH data/pool/part$k > data/own/relabel$n.log 2>&1
  echo "$(date '+%F %T') iteration $n: $(wc -l < data/own/own$n.txt) labels, training" >> $LOG
  ./build/bgtrain build data/own_iter$n.bin data/own/own*.txt >> $LOG 2>&1
  ./build/bgtrain train data/own_iter$n.bin nets/iter$n.bgnn --epochs 12 --threads $TH --hidden 512 --l2 64 --l3 64 \
    --lr 3e-3 > data/own/train$n.log 2>&1
  grep "ep 12\]" data/own/train$n.log >> $LOG
  for p in 1 2; do ./build/bgcli bench data/bench/bench1.txt $p nets/iter$n.bgnn $TH >> $LOG; done
  ./build/bgcli duel nets/iter$n.bgnn $BEST 20000 2 2 $TH > data/own/duel$n.txt
  cat data/own/duel$n.txt >> $LOG
  PPG=$(sed -n 's/.*A ppg \([-+0-9.]*\).*/\1/p' data/own/duel$n.txt)
  if awk "BEGIN{exit !($PPG > 0)}"; then
    BEST=nets/iter$n.bgnn
    echo "$(date '+%F %T') iteration $n: adopted $BEST (ppg $PPG)" >> $LOG
  else
    echo "$(date '+%F %T') iteration $n: kept $BEST (ppg $PPG)" >> $LOG
  fi
  k=$((k + 1))
done
echo "$(date '+%F %T') done, best=$BEST" >> $LOG
