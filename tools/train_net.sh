#!/bin/sh
# Builds the dataset from all label files and trains a network, then benchmarks it.
#   tools/train_net.sh <name> [hidden] [l2] [l3] [epochs] [threads]
# Result: nets/<name>.bgnn, log in data/<name>.log
set -e
NAME=${1:?name}
HID=${2:-256}; L2=${3:-96}; L3=${4:-64}; EP=${5:-12}; TH=${6:-8}
./build/bgtrain build data/$NAME.bin data/labels/*.txt
./build/bgtrain train data/$NAME.bin nets/$NAME.bgnn --epochs $EP --threads $TH --hidden $HID --l2 $L2 --l3 $L3 --lr 3e-3 \
  > data/$NAME.log 2>&1
grep val data/$NAME.log | tail -1
for p in 1 2; do ./build/bgcli bench data/bench/bench1.txt $p nets/$NAME.bgnn $TH; done
