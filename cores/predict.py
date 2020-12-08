import os
import yaml

from cores.config import CONFIG_MAP
from cores.logger import init_logger, logger, seed_everything


def predict(config):
    # Read YAML file
    with open(config, 'r') as stream:
        config_loaded = yaml.safe_load(stream)
        opts = CONFIG_MAP[config_loaded['model_type']]()
        opts.add_attribute(config_loaded)

    if not os.path.exists(opts.saved_dir):
        os.makedirs(opts.saved_dir)
    init_logger(log_file=opts.saved_dir + '/{}_predict.log'.format(opts.model_type))
    seed_everything(opts.random_seed)
    
    return None