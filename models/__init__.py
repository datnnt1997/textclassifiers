import os
import torch

from .fasttext import FastText
from .textrnn import TextRNN
from cores.logger import logger

MODEL_MAP = {
    'fasttext': FastText,
    'textrnn': TextRNN
}


def save_checkpoint(save_dir, model, epoch, train_avg_loss, train_acc_score, train_f1_score,
                    eval_loss, eval_acc, eval_macro_f1, metric):
    file_model = os.path.join(save_dir, "model.model")
    file_infor = os.path.join(save_dir, "summary.txt")
    torch.save(model.state_dict(), file_model)
    f = open(file_infor, 'w', encoding="utf-8")
    f.write("=" * 30 + "STATISTITCALS" + "=" * 30 + "\n")
    f.write("File model: {}\n".format(file_model))
    f.write("Epoch: {}\n".format(epoch))
    f.write("Train: \n")
    f.write("\tLoss: {}\n".format(train_avg_loss))
    f.write("\tAccuracy: {}\n".format(train_acc_score))
    f.write("\tF1 score: {}\n".format(train_f1_score))
    f.write("Evaluation: \n")
    f.write("\tLoss: {}\n".format(eval_loss))
    f.write("\tAccuracy: {}\n".format(eval_acc))
    f.write("\tF1 score: {}\n".format(eval_macro_f1))
    f.write("=" * 30 + "CLASS METRIC" + "=" * 30 + "\n")
    f.write(metric + "\n")
    f.close()
    logger.info("saved model at epoch %d at %s" % (epoch, str(file_model)))


__all__ = ['FastText', 'TextRNN', 'MODEL_MAP', 'save_checkpoint']