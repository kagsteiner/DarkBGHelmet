#!/bin/sh
# Starts N background gnubg labeling workers: tools/label_run.sh <workers> <labels-per-worker> [plies] [seed-base]
N=${1:-14}; COUNT=${2:-1000000}; PLIES=${3:-2}; BASE=${4:-100}
mkdir -p data/labels
for i in $(seq 1 $N); do
  BG_SEED=$((BASE+i)) BG_OUT=data/labels/g${PLIES}_s$((BASE+i)).txt BG_PLIES=$PLIES BG_COUNT=$COUNT \
    nohup gnubg -t -q -p tools/gnubg_label.py > /dev/null 2>&1 &
done
echo "started $N workers"
