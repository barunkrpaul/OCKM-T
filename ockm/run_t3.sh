#!/bin/bash
cd "$(dirname "$0")"
R=../results/development
mkdir -p $R
python3 -W ignore bench_t3.py --datasets bench --out $R/runs.csv
python3 -W ignore bench_t3.py --datasets MNIST,FMNIST --out $R/runs.csv
echo ALL_DONE
