#!/bin/bash

set -eu

export PYTHONPATH=google-research/

basedir=$1
max_seq_length=$2

for DIR in $( find ${basedir}/wiki/ -mindepth 1 -type d ); do
  out=${DIR}/all-maxseq${max_seq_length}.tfrecord
  if [ -f ${out} ]; then
    continue
  fi
  python ALBERT/create_pretraining_data.py \
    --input_file=${DIR}/all.txt \
    --output_file=${out} \
    --spm_model_file=${basedir}/model/wiki-ja.model \
    --vocab_file=${basedir}/model/wiki-ja.vocab \
    --do_lower_case=True \
    --max_seq_length=${max_seq_length} \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 \
    --do_whole_word_mask=False \
    --do_permutation=False \
    --favor_shorter_ngram=False \
    --random_next_sentence=False
done
