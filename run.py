import os
import argparse
from cores import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, required=True, help="The config file.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError("Config file ({}) not exists or is empty !!!".format(args.config))
    train(args.config)

