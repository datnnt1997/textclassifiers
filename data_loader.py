import os

from torchtext.data import Field, Dataset, Example, Iterator


class DataLoader:
    def __init__(self, datapath: str):
        assert os.path.exists(datapath), f"{datapath} is not exists!!!"
        self.datapath = datapath

        self.batch_size = 32
        self.device = "cuda"
        self.max_seq_len = 256

        self.fields = self.get_fields()

    @staticmethod
    def get_fields():
        text_field = Field(tokenize=lambda x: x.split(), include_lengths=True, batch_first=True)
        label_field = Field(sequential=False, is_target=True)
        return [('text', text_field), ('label', label_field)]

    def read_data(self):
        examples = []
        with open(self.datapath, "r", encoding="utf-8") as reader:
            for line in reader.readlines():
                title, content, default, category, provinces = line.strip().split("\t")
                examples.append(Example.fromlist([content, category], self.fields))
            reader.close()
        return Dataset(examples, self.fields)

    def iter_dataset(self, dataset: Dataset, is_train: bool = True) -> Iterator:
        cur_iter = Iterator(
            dataset=dataset,
            batch_size=self.batch_size,
            device=self.device,
            batch_size_fn=None,
            train=is_train,
            repeat=False,
            shuffle=True,
            sort=False,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
        )
        return cur_iter
