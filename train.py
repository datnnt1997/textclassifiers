# import warnings
# warnings.simplefilter("ignore", UserWarning)

import torch

from models.reg_lstm import RegLSTM
from models.lstm import LSTM
from data_loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

CONFIG = {
    # Training parameters
    "lr": 0.01,
    'epochs': 100,
    'batch_size': 64,
    "weight_decay": 0,
    'epoch_decay': 15,
    'max_seq_len': 512,
    'data_path': 'dataset/output.tsv',
    'save_path': 'outputs/',
    'device': None,  # ['cpu', 'cuda', None]

    # Model parameters
    'bidirectional': True,
    'bottleneck_layer': False,
    'num_layers': 1,
    'hidden_dim': 256,
    'mode': 'rand',  # ['rand', 'static', 'non-static']
    'words_num': 300,
    'embed_dim': 300,
    'dropout': 0.5,
    'wdrop': 0.0,
    'beta_ema': 0.99,
    'embed_droprate': 0.0,
    'tar': 0.0,
    'ar': 0.0,


}


def main():
    if CONFIG['device'] is None:
        CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(CONFIG)
    train_iter, test_iter = data_loader.read_and_iter(test_ratio=0.2,
                                                      is_build_vocab=True,
                                                      special_tokens=['num', 'url'])

    CONFIG['words_num'] = len(data_loader.text_field.vocab)
    CONFIG['target_class'] = len(data_loader.label_field.vocab)
    CONFIG['pad_idx'] = data_loader.pad_idx

    model = LSTM(CONFIG, vectors=data_loader.text_field.vocab.vectors)
    print(model)
    parameter = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameter, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss().cuda() if CONFIG['device'] == 'cuda' else torch.nn.CrossEntropyLoss()
    for epoch in range(CONFIG['epochs']):
        print(f"{'='*25}Epoch {epoch}{'='*25}")
        train_iter.init_epoch()
        model.train()
        avg_loss = 0
        y_true = []
        y_pred = []
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            texts, lengths = batch.text
            labels = batch.label
            if hasattr(model, 'tar') and model.tar:
                prods, scores, rnn_outs = model(texts, lengths)
            else:
                prods, scores = model(texts, lengths)
            loss = criterion(prods, labels)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            y_true += labels.tolist()
            y_pred += [y.argmax().item() for y in prods]
        print(classification_report(y_true, y_pred, target_names=data_loader.label_field.vocab.itos))
        print(f"AVG Loss = {avg_loss}")


if __name__ == "__main__":
    main()
