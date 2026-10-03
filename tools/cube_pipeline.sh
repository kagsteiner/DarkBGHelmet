#!/bin/sh
# Waits for the cube labelling to finish, trains the cube head on iter1 and
# compares it with the formula in cubeful duels. Log: data/cube/pipeline.log
cd "$(dirname "$0")/.."
LOG=data/cube/pipeline.log
while pgrep -f "bgtrain cubelabel" > /dev/null; do sleep 60; done
echo "$(date '+%F %T') labelling done: $(wc -l < data/cube/cube4ply.txt) positions" >> $LOG
./build/bgtrain traincube nets/iter1.bgnn nets/iter1_cube.bgnn --epochs 30 data/cube/cube4ply.txt > data/cube/traincube.log 2>&1
grep -E "init|ep 30\]|written" data/cube/traincube.log >> $LOG
for P in 1 2; do
  G=200000; [ $P -eq 2 ] && G=40000
  BG_DUEL_CUBE=1 BG_B_NOHEAD=1 ./build/bgcli duel nets/iter1_cube.bgnn nets/iter1_cube.bgnn $G $P $P 17 >> $LOG
done
BG_DUEL_CUBE=1 BG_JACOBY=0 BG_B_NOHEAD=1 ./build/bgcli duel nets/iter1_cube.bgnn nets/iter1_cube.bgnn 200000 1 1 17 >> $LOG
echo "$(date '+%F %T') done" >> $LOG
