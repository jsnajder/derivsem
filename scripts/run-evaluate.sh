#!/bin/bash

for model in "baseline" "add" "lexfun"; do
  for space in "cbow-w2" "cbow-w5" "cbow-w10"; do
    echo "nohup python -u ../src/evaluate.py ../data/pairs-all.txt $model $space ../results/ > ../results/$model-$space.txt &"
  done;
done;

