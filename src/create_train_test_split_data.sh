#!/bin/bash

set -eu

export PYTHONPATH=`dirname $0`/..

basedir=$1

for file in $( find ${basedir} -mindepth 1 -name "*.json" ); do
  python src/train_test_split_data.py \
    --input_file=${file} 
done
