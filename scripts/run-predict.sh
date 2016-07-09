#!/bin/bash

# ./run-predict.sh [baseline|add|lexfun]

model=$1
space="cbow-w2"

for pat in "00" "01" "02" "03" "04" "05"; do
  echo python -u ../src/predict.py ../data/partitioned-pairs.csv $1 $space ../data/patterns.txt-$pat ../results/$model-$space-$pat.txt > ../results/$model-$space-$pat.log &

