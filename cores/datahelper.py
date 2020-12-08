import os
import torch
import multiprocessing

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cores.logger import logger


def read_data(file_path, delimiter='\t'):
    datasets = []
    if delimiter == '\\t':
        delimiter = '\t'
    with open(file_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        for row in rows:
            cols = row.split(delimiter)
            if len(cols) != 0:
                datasets.append(cols)
    return datasets


def load_word2vec(opts):
    with open(opts.pretrained_embedding, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vocab_size = len(lines)
        embed_dim = len(lines[0].split(' ')) - 1
        vocab = []
        vectors = []
        if opts.pad_token is not None:
            vocab.append(opts.pad_token)
            vocab_size += 1
            vectors.append(torch.zeros([embed_dim], dtype=torch.float))
        if opts.pad_token is not None:
            vocab.append(opts.unk_token)
            vocab_size += 1
            vectors.append(torch.rand([embed_dim], dtype=torch.float))
        for line in tqdm(lines, total=vocab_size, leave=False, position=0):
            line = line.split(' ')
            token_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            vocab.append(line[0])
            vectors.append(torch.tensor(token_vector, dtype=torch.float))
    return vocab, torch.stack(vectors)


class Example(object):
    def __init__(self, input_ids: list, label_id: int, seq_len: int, raw_text: str = None, raw_label: str = None):
        self.input_ids = input_ids
        self.label_id = label_id
        self.seq_len = seq_len
        self.raw_text = raw_text
        self.raw_label = raw_label

    def __str__(self):
        ex_str = f"""Input IDs: {self.input_ids}\n
                     Label ID: {self.label_id}\n
                     Sequence Length: {self.seq_len}\n
        """
        if self.raw_text is not None:
            ex_str += f"Raw Text: {self.raw_text}\n"
        if self.raw_label is not None:
            ex_str += f"Label: {self.raw_label}\n"
        return ex_str


class TextDataset(Dataset):
    def __init__(self, file_path, model_type: str, data_format: list = ['text', 'label'], delimiter='\t', vocab=None,
                 label_set=None, max_len=256, pad_token="<pad>",  unk_token="<unk>"):
        self.file_path = file_path
        self.model_type = model_type
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_len = max_len
        self.data_format = data_format

        self.init_vocal, self.vocab = (True, [self.pad_token, self.unk_token]) if vocab is None else (False, vocab)
        self.init_label, self.label_set = (True, []) if label_set is None else (False, label_set)

        dataset_name = os.path.basename(self.file_path).split('.')[0].strip()
        cached_dir = os.path.dirname(self.file_path)
        self.cached_file = cached_dir + f'/cached_file_{self.model_type}_{self.max_len}_{dataset_name}'
        if os.path.exists(self.cached_file):
            self.examples = []
            self.load_cached_file()
        else:
            self.examples = self.create_examples(read_data(file_path, delimiter))
            self.cache_dataset()

    @property
    def pad_id(self):
        return self.vocab.index(self.pad_token)

    @property
    def unk_id(self):
        return self.vocab.index(self.unk_token)

    def convert_tokens_to_ids(self, text):
        tokens = text.split()
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab.index(token))
            else:
                if self.init_vocal:
                    self.vocab.append(token)
                    token_ids.append(len(self.vocab)-1)
                else:
                    token_ids.append(self.vocab.index(self.unk_token))
        return token_ids

    def convert_label_to_id(self, label):
        if label in self.label_set:
            return self.label_set.index(label)
        else:
            if self.init_label:
                self.label_set.append(label)
                return len(self.label_set) - 1
            else:
                raise Exception(f"Label {label} is not found !!")

    def preprocess(self, text):
        return text

    def cache_dataset(self):
        data = {
            'vocab': self.vocab,
            'label_set': self.label_set,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'max_len': self.max_len,
            'data_format': self.data_format,
            'examples': self.examples
        }
        logger.info("\tSaving Dataset into cached file %s", self.cached_file)
        torch.save(data, self.cached_file)

    def load_cached_file(self):
        logger.info("\tLoad Dataset from cached file %s", self.cached_file)
        data = torch.load(self.cached_file)
        for key, value in data.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    def create_example(self, cols, text_id, label_id, pad_id):
        raw_text = cols[text_id].strip()
        raw_label = cols[label_id].strip()

        ex_text = self.preprocess(raw_text)
        ex_input_ids = self.convert_tokens_to_ids(ex_text)
        ex_label_id = self.convert_label_to_id(raw_label)
        ex_length = len(ex_input_ids)
        if ex_length < self.max_len:
            pad_ids = [pad_id] * (self.max_len - ex_length)
            ex_input_ids.extend(pad_ids)
        elif ex_length > self.max_len:
            ex_input_ids = ex_input_ids[:self.max_len]
            ex_length = self.max_len

        return Example(input_ids=ex_input_ids,
                       label_id=ex_label_id,
                       seq_len=ex_length,
                       raw_text=raw_text,
                       raw_label=raw_label)

    def create_examples(self, dataset):
        examples = []
        pad_id = self.vocab.index(self.pad_token)
        text_id = self.data_format.index('text')
        label_id = self.data_format.index('label')
        with multiprocessing.pool.ThreadPool(processes=4) as pool:
            params = list(map(lambda cols: (cols, text_id, label_id, pad_id), dataset))
            examples = tqdm(pool.starmap(self.create_example, params))
        # for cols in tqdm(dataset):
        #     raw_text = cols[text_id].strip()
        #     raw_label = cols[label_id].strip()
        #
        #     ex_text = self.preprocess(raw_text)
        #     ex_input_ids = self.convert_tokens_to_ids(ex_text)
        #     ex_label_id = self.convert_label_to_id(raw_label)
        #     ex_length = len(ex_input_ids)
        #     if ex_length < self.max_len:
        #         pad_ids = [pad_id] * (self.max_len - ex_length)
        #         ex_input_ids.extend(pad_ids)
        #     examples.append(Example(input_ids=ex_input_ids,
        #                             label_id=ex_label_id,
        #                             seq_len=ex_length,
        #                             raw_text=raw_text,
        #                             raw_label=raw_label))

        return examples

    @staticmethod
    def collate_fn(batch):
        all_input_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
        all_lens, indices = torch.sort(all_lens, descending=True)
        all_input_ids = all_input_ids[indices][:, :all_lens[0]]
        all_labels = all_labels[indices]
        return all_input_ids, all_lens, all_labels

    @staticmethod
    def transformer_collate_fn(batch):
        all_input_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
        return all_input_ids, all_lens, all_labels

    def get_collate_fn(self):
       if self.model_type in ['transformer', 'bert']:
           return self.transformer_collate_fn
       else:
           return self.collate_fn

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        ex_input_tensor = torch.tensor(example.input_ids, dtype=torch.long)
        ex_label_tensor = torch.tensor(example.label_id, dtype=torch.long)
        ex_seq_length_tensor = torch.tensor(example.seq_len, dtype=torch.long)
        return ex_input_tensor, ex_seq_length_tensor, ex_label_tensor


if __name__ == "__main__":
    dataset = TextDataset("dataset/query_wellformedness/dev.txt")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch)
