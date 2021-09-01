#!/usr/bin/env python3
import argparse
import time
import os
import gc
import random
import warnings
import glob
import json
import math
import logging
import argparse
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq
from absl import app, flags, logging
from collections import defaultdict
from pathlib import Path

train_test_split_rate = 0.3

# basic setitng
FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', default=None, help='Input file')

def config():    
    main_path = Path("/home/jupyter/gogolook")
    main_data_path = main_path / 'data' / 'jp_data'
    wiki_preprocess_data_path = main_data_path / 'preprocessing_data'
    train_data_path = main_data_path / 'train_pretraining_data'
    valid_data_path = main_data_path / 'valid_pretraining_data'

    for path in [train_data_path, valid_data_path]:
        if not os.path.exists(str(path)):
            logging.info(f'Create folder - {path}')
            os.mkdir(str(path))
    return train_data_path, valid_data_path
  
def convert_to_parquet(data, save_to_disk_path):
    data_list = [data[i] for i in data.keys()]
    temp_df = pd.DataFrame.from_dict(data_list)
    temp_table = pa.Table.from_pandas(temp_df)
    pq.write_table(temp_table, save_to_disk_path)

def train_test_split(file, train_data_path, valid_data_path):
    logging.info("Start to processing...")
    file_basename = os.path.basename(file)
    train_file_path = train_data_path / ('train_' + file_basename.split('.')[0] + '.parquet')
    valid_file_path = valid_data_path / ('valid_' + file_basename.split('.')[0] + '.parquet')
    
    valid_data_dict = {}
    with open(file, 'r') as f:
        train_data_dict = json.load(f)
                 
    total_number = len(train_data_dict)
    sample_index_valid = random.sample(range(total_number), math.floor(total_number * train_test_split_rate))
    logging.info(f"File: {file_basename} - total number: {total_number}")
                             
    for k, v in enumerate(sample_index_valid):
        valid_data_dict[k] = train_data_dict.pop(str(v))
        valid_data_dict[k]['original_key'] = v
                 
    #with open(train_file_path, 'w') as train_file:
    #    json.dump(preprocess_data_dict, train_file)
        
    #with open(valid_file_path, 'w') as valid_file:
    #    json.dump(valid_data_dict, valid_file)
     
    convert_to_parquet(train_data_dict, train_file_path)
    convert_to_parquet(valid_data_dict, valid_file_path)
    
    logging.info(f"Number of train data: {len(train_data_dict)} - Number of valid data: {len(valid_data_dict)}")
    del train_data_dict, valid_data_dict
    gc.collect()

    
def main(argv):
    start_time = time.time()
    train_dir, valid_dir = config()
    train_test_split(FLAGS.input_file, train_dir, valid_dir)
    logging.info(f"Total files processing: {time.time() - start_time}")
    
if __name__ == "__main__":
    flags.mark_flags_as_required(['input_file'])
    app.run(main)