import os

from torch.utils.data import DataLoader
from config import *
from datahelper import TextDataset
from models import FastText
from sklearn import metrics
from tqdm import tqdm


def save_checkpoint(save_dir, model, epoch, loss, acc, macro_f1):
    file_model = os.path.join(save_dir, "model.model")
    file_infor = os.path.join(save_dir, "summary.txt")
    print("saving %s" % file_model)
    torch.save(model.state_dict(), file_model)
    f = open(file_infor, 'w', encoding="utf-8")
    f.write("=" * 30 + "STATISTITCALS" + "=" * 30)
    f.write("File model: {}\n".format(file_model))
    f.write("Epoch: {}\n".format(epoch))
    f.write("Loss: {}\n".format(loss))
    f.write("Evaluation: \n")
    f.write("Accuracy: {}\n".format(acc))
    f.write("F1 score: {}\n".format(macro_f1))
    f.write("=" * 30 + "CLASS METRIC" + "=" * 30)
    f.close()
    print("saved model at epoch %d" % epoch)


def eval(test_iter, model, device):
    avg_loss = 0
    predicts = []
    actuals = []
    model.eval()
    tqdm_bar = tqdm(enumerate(test_iter), total=len(test_iter), desc="Valid")
    for idx, batch in tqdm_bar:
        sent, sent_lens, labels = batch
        input_ids, seq_lens, label_ids = batch
        if device == 'cuda':
            input_ids = input_ids.to(device)
            seq_lens = seq_lens.to(device)
            label_ids = label_ids.to(device)
        loss, probs = model(input_ids, seq_lens, label_ids)

        avg_loss += loss.item()
        predicts += [y.argmax().item() for y in probs]
        actuals += labels.tolist()

    metric = metrics.classification_report(actuals, predicts)
    acc_score = metrics.accuracy_score(actuals, predicts)
    macro_f1_score = metrics.f1_score(actuals, predicts, average="macro")
    print(metric)
    print("VALID Macro F1 score: " + str(macro_f1_score))
    print("VALID Accurancy score: " + str(acc_score))
    print("VALID LOSS: {}".format(avg_loss / len(test_iter)))
    return acc_score, macro_f1_score, avg_loss


def main():
    opts = Config()

    if not os.path.exists(opts.saved_dir):
        os.makedirs(opts.saved_dir)
    print("Loading TRAIN dataset ...")
    train_dataset = TextDataset(opts.train_path, data_format=['text', 'label'],
                                delimiter='\t', vocab=None, label_set=None, max_len=256,
                                pad_token="<pad>", unk_token="<unk>")
    print("Loading TEST dataset ...")

    test_dataset = TextDataset(opts.test_path, data_format=['text', 'label'],
                               delimiter='\t', vocab=train_dataset.vocab, label_set=train_dataset.label_set,
                               max_len=256, pad_token="<pad>", unk_token="<unk>")

    model = FastText(vocab_size=len(train_dataset.vocab),
                     embed_dim=opts.embed_dim,
                     hidden_dim=opts.hidden_dim,
                     num_labels=len(train_dataset.label_set),
                     vectors=None,
                     pad_idx=train_dataset.pad_id)

    if opts.pretrained_model_dir is not None:
        model_checkpoint = torch.load(opts.pretrained_model_dir + "/model.model")
        model.load_state_dict(model_checkpoint)

    print("=" * 30 + "MODEL SUMMARY" + "=" * 30)
    print(model)
    print("=" * 73)

    if opts.device == 'cuda':
        model.cuda()

    if opts.optim == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr)
    elif opts.optim == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opts.lr, momentum=opts.momentum)
    else:
        raise Exception(f"{opts.optim} is not Found !!")

    best_score = float('-inf')
    train_iter = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,
                            collate_fn=train_dataset.collate_fn)

    valid_iter = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True,
                            collate_fn=train_dataset.collate_fn)

    for epoch in range(opts.num_epoch):
        print(f"Epoch: {epoch}/{opts.num_epoch}")
        epoch_avg_loss = 0
        predicts = []
        actuals = []
        model.train()
        model.to(opts.device)
        tqdm_bar = tqdm(enumerate(train_iter), total=len(train_iter), desc="Train")
        for idx, batch in tqdm_bar:
            optimizer.zero_grad()

            input_ids, seq_lens, label_ids = batch
            if opts.device == 'cuda':
                input_ids = input_ids.to(opts.device)
                seq_lens = seq_lens.to(opts.device)
                label_ids = label_ids.to(opts.device)
            loss, probs = model(input_ids, seq_lens, label_ids)
            loss.backward()
            optimizer.step()

            epoch_avg_loss += loss.item()
            predicts += [y.argmax().item() for y in probs]
            actuals += label_ids.tolist()
        print(epoch_avg_loss)
        acc_score = metrics.accuracy_score(actuals, predicts)
        macro_f1_score = metrics.f1_score(actuals, predicts, average="macro")
        print("TRAIN Macro F1 score: " + str(macro_f1_score))
        print("TRAIN Accurancy score: " + str(acc_score))
        print("TRAIN LOSS: {}".format(epoch_avg_loss / len(train_iter)))

        if epoch % opts.valid_interval == 0:
            acc_score, macro_f1_score, avg_loss = eval(valid_iter, model, opts.device)
            if macro_f1_score > best_score:
                save_checkpoint(opts.saved_dir, model, epoch, avg_loss,  acc_score, macro_f1_score)
                best_score = macro_f1_score
    return best_score


if __name__ == "__main__":
    main()
