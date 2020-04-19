#!/usr/bin/env python3

import configparser
import glob
import os
import sentencepiece as sp

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

TEXTDIR = config['DATA']['TEXTDIR']
PREFIX = config['SENTENCEPIECE']['PREFIX']
VOCABSIZE = config['SENTENCEPIECE']['VOCABSIZE']
CTLSYMBOLS = config['SENTENCEPIECE']['CTLSYMBOLS']


def _get_text_file(text_dir=TEXTDIR):
    file_list = glob.glob(f'{text_dir}/**/*.txt')
    files = ",".join(file_list)
    return files


def train(prefix=PREFIX, vocab_size=VOCABSIZE, ctl_symbols=CTLSYMBOLS):
    files = _get_text_file()
    # https://github.com/google-research/albert/blob/a41cf11700c1ed2b7beab0a2649817fa52c8d6e1/README.md#sentencepiece
    command = f'--input={files} --model_prefix={prefix} --vocab_size={vocab_size} ' \
              f'--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 --user_defined_symbols=(,),",-,.,–,£,€ ' \
              f'--control_symbols={ctl_symbols} --input_sentence_size=15000000 ' \
              f'--shuffle_input_sentence=true --character_coverage=0.99995 --model_type=unigram'
    sp.SentencePieceTrainer.Train(command)


def main():
    train()


if __name__ == "__main__":
    main()
