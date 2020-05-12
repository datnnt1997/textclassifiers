import os

from torchtext.data import Field, LabelField, Dataset, Example, BucketIterator
from torchtext.vocab import Vectors
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        assert os.path.exists(config['data_path']), f"{config['data_path']} is not exists!!!"
        self.data_path = config['data_path']

        self.batch_size = config['batch_size']
        self.device = config['device']
        self.max_seq_len = config['max_seq_len']

        self.text_field, self.label_field = self.get_fields()

        self.pad_token = self.text_field.pad_token
        self.pad_idx = None

    @staticmethod
    def get_fields() -> (Field, LabelField):
        text_field = Field(tokenize=lambda x: x.split(), include_lengths=True, batch_first=True)
        label_field = LabelField()
        return text_field, label_field

    def read_data(self,
                  test_ratio: float = 0.2,
                  is_build_vocab: bool = True,
                  special_tokens: list = []) -> (Dataset, Dataset):
        examples = []
        fields = [('text', self.text_field), ('label', self.label_field)]
        with open(self.data_path, "r", encoding="utf-8") as reader:
            for line in reader.readlines():
                content, category = line.split("\t")
                if len(content.split()) <= self.max_seq_len:
                    examples.append(Example.fromlist([content, category.strip()], fields))
                else:
                    truncated_content = " ".join(content.split()[:self.max_seq_len])
                    examples.append(Example.fromlist([truncated_content, category.strip()], fields))
            reader.close()
        train_examples, test_examples = train_test_split(examples, test_size=test_ratio, random_state=42, shuffle=True)
        train_dataset = Dataset(train_examples, fields)
        test_dataset = Dataset(test_examples, fields)
        if is_build_vocab:
            vectors = Vectors(name="embeding.emb", cache="/")
            self.text_field.build_vocab(train_dataset, test_dataset, specials=special_tokens, vectors=vectors)
            #self.text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
            self.label_field.build_vocab(train_dataset, test_dataset)
            self.pad_idx = self.text_field.vocab.stoi[self.pad_token]
        else:
            self.text_field.build_vocab(train_dataset, test_dataset, specials=special_tokens)
            self.label_field.build_vocab(train_dataset, test_dataset)
            self.pad_idx = self.text_field.vocab.stoi[self.pad_token]
        return train_dataset, test_dataset

    def iter_dataset(self, dataset: Dataset, is_train: bool = True) -> BucketIterator:
        cur_iter = BucketIterator(
            dataset=dataset,
            batch_size=self.batch_size,
            device=self.device,
            batch_size_fn=None,
            train=is_train,
            repeat=False,
            shuffle=True,
            sort=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.text),
        )
        return cur_iter

    def read_and_iter(self, test_ratio: float = 0.2,
                      is_build_vocab: bool = True,
                      special_tokens: list = []) -> (BucketIterator, BucketIterator):
        train_dataset, test_dataset = self.read_data(test_ratio, is_build_vocab, special_tokens)
        train_iter = self.iter_dataset(train_dataset)
        test_iter = self.iter_dataset(test_dataset, is_train=False)
        return train_iter, test_iter
