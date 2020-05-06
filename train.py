import torch

from models.reg_lstm import RegLSTM
from data_loader import DataLoader
CONFIG = {
    # Training parameters
    "lr": 0.01,
    'epochs': 30,
    'batch_size': 32,
    "weight_decay": 0,
    'epoch_decay': 15,
    'max_seq_len': 512,
    'save-path': 'outputs/',
    'device': None,  # ['cpu', 'cuda', None]

    # Model parameters
    'bidirectional': True,
    'bottleneck_layer': False,
    'num_layers': 1,
    'hidden_dim': 512,
    'mode': 'rand',  # ['rand', 'static', 'non-static']
    'words_dim': 300,
    'embed_dim': 300,
    'dropout': 0.5,
    'wdrop': 0.1,
    'beta_ema': 0.99,
    'embed_droprate': 0.2,
    'tar': 0.0,
    'ar': 0.0,


}


def main():
    if CONFIG['device'] is None:
        CONFIG['device'] = torch.cuda.current_device()
    data_loader = DataLoader(CONFIG)
    train_iter, test_iter = data_loader.read_and_iter(test_ratio=0.2,
                                                      is_build_vocab=True,
                                                      special_tokens=['NUM', 'URL'])
    model = RegLSTM(CONFIG)
    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    print()

if __name__ == "__main__":
    main()
