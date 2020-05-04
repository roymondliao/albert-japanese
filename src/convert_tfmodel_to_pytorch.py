import json
import logging

from transformers import AlbertConfig, AlbertForMaskedLM, load_tf_weights_in_albert, AlbertTokenizer

logging.basicConfig(level=logging.DEBUG)


def main(args):
    with open(args.config) as fp:
        data = json.loads(fp.read())
    config = AlbertConfig(**data)
    model = AlbertForMaskedLM(config)
    model: AlbertForMaskedLM = load_tf_weights_in_albert(model, config, args.checkpoint)
    model.save_pretrained(args.output)

    tokenizer = AlbertTokenizer.from_pretrained(args.spiece, keep_accents=True)
    tokenizer.save_pretrained(args.output)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("spiece")
    parser.add_argument("output")
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
