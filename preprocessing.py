import re
import string
import gensim

from os import path
from underthesea import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

from tqdm import tqdm

URL_PATTERN = ".[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2," \
            "}|www\.[a-zA-Z0-9]+\.[^\s]{2,} "

NUMBER_PATTERN = "([0-9]+[.,\)/:]?([0-9]+)?[.,\/):]?)"
DATASET = "dataset/Dataset_gopy.tsv"
# OUT = "dataset/preprocess_dataset.tsv"
OUT = "dataset/"

example_content = """Trước tiên tôi xin cảm ơn cơ quan chức năng đã làm việc và phản hồi về vấn đề đỗ xe lấn chiếm lòng đường tại K96 H17/11 Hải Hồ (https://gopy.danang.gov.vn/gop-y?pageid=view&ykien=23557)
Trong bài phản ánh trước đã được phản hồi là Công an phường làm việc với các hộ dân trong tổ 27 và 28 để tránh đỗ xe ở lòng đường kiệt gây cản trở giao thông. Tuy nhiên không hiểu sao gia đình bà Lan vẫn tiếp tục tái diễn việc đỗ xe tại nơi ngã ba này cả ngày lẫn đêm, gây khó khăn cho việc đi lại của người dân trong khu phố. Theo như phản hồi từ Công an phường thì do đường kiệt không có bảng cấm đỗ nên không có cơ sở để xử lý, nhưng theo Khoản 4 Điều 18 Luật giao thông đường bộ quy định về Dừng xe, đỗ xe trên đường bộ:
"4. Người điều khiển phương tiện không được dừng xe, đỗ xe tại các vị trí sau đây
e) Nơi đường giao nhau và trong phạm vi 5 mét tính từ mép đường giao nhau"
Vậy kính mong cơ quan chức năng xem xét việc đỗ xe này có đúng theo quy định hay không? Và đề nghị gia đình bà Lan gửi xe vào bãi nếu như không có nhu cầu sử dụng thường xuyên để trả lại lối đi chung cho người dân khu phố.
Ngoài ra cũng xin thông tin thêm là hiện nay tại K96 H17 Hải Hồ có nhiều hộ dân sử dụng xe ô tô và đỗ trước nhà nên với đường kiệt chỉ 4m và là đường thông để ra Lương Ngọc Quyến thì việc giao thông không còn thông thoáng như trước. Kính đề nghị cơ quan chức năng xem xét về việc gắn biển cấm đỗ lại đường kiệt này nếu hợp lý.
Xin trân trọng cám ơn."""


def statistic(dataset_path: path):
    assert path.exists(dataset_path), f"{dataset_path} is not exists!"
    statis = {"max_length": 0,
              "content": "",
              "lengths": {},
              "sentence_lengths": {},
              "max_sentence_length": 0,
              "max_stence_length_content": ""}
    with open(dataset_path, "r", encoding="utf-8") as reader:
        examples = reader.readlines()
        for example in examples[1:]:
            title, content, default, category, provinces = example.split("\t")
            example_lenght = len(content.split())
            example_sentences = sent_tokenize(content)

            if example_lenght not in statis["lengths"]:
                statis["lengths"][example_lenght] = 1
            else:
                statis["lengths"][example_lenght] += 1
            if example_lenght > statis["max_length"]:
                statis["max_length"] = example_lenght
                statis["content"] = content

            if len(example_sentences) not in statis["sentence_lengths"]:
                statis["sentence_lengths"][len(example_sentences)] = 1
            else:
                statis["sentence_lengths"][len(example_sentences)] += 1
            if len(example_sentences) > statis["max_sentence_length"]:
                statis["max_sentence_length"] = len(example_sentences)
                statis["max_stence_length_content"] = content
    print("="*50)
    print("Example lengths:")
    print(statis["lengths"])
    print("=" * 50)
    print("Max example leg:")
    print(statis["max_length"])
    print("=" * 50)
    print("Longest Example content:")
    print(statis["content"])

    print("=" * 50)
    print("Num of example sentences:")
    print(statis["sentence_lengths"])
    print("=" * 50)
    print("Max num of sentence:")
    print(statis["max_sentence_length"])
    print("=" * 50)
    print("Sentence Example content:")
    print(statis["max_stence_length_content"])


def text_normalize(text):
    text = text.strip()
    #norm_word_tokenize = word_tokenize(text, format='text')
    norm_url = re.sub(URL_PATTERN, " URL ", text)
    #norm_number = re.sub(NUMBER_PATTERN, " NUM ", norm_url)
    #norm_punc = re.sub(f"[{string.punctuation}]", " ", norm_number)
    #norm_newline = re.sub("\n+| +", " ", norm_url)
    return norm_url


def preprocess(dataset_path, out_path):
    assert path.exists(dataset_path), f"{dataset_path} is not exists!"
    with open(dataset_path, "r", encoding="utf-8") as reader, \
            open(out_path, "w", encoding="utf-8") as writer:
        examples = reader.readlines()
        for example in tqdm(examples[1:]):
            title, content, default, category, provinces, _ = example.split("\t")
            writer.write(f"{text_normalize(content)}\t{category}\n")
        writer.close()
        reader.close()


def preprocess_split_data(dataset_path, out_path):
    assert path.exists(dataset_path), f"{dataset_path} is not exists!"
    with open(dataset_path, "r", encoding="utf-8") as reader, \
            open(out_path+"/train.tsv", "w", encoding="utf-8") as train_writer, \
            open(out_path+"/test.tsv", "w", encoding="utf-8") as test_writer:
        examples = reader.readlines()
        data = []
        for example in tqdm(examples[1:]):
            title, content, default, category, provinces, _ = example.split("\t")
            data.append(f"{text_normalize(content)}\t{category}\n")
        train_examples, test_examples = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
        for example in train_examples:
            train_writer.write(example)
        for example in test_examples:
            test_writer.write(example)
        train_writer.close()
        test_writer.close()
        reader.close()


def build_word2vec():
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format("baomoi.vn.model.bin", binary=True)
    wv.save_word2vec_format("embeding.emb")
    print()

def load_vectors():
    from torchtext.vocab import Vectors
    vectors = Vectors(name="embeding.emb", cache="/")
    print()

if __name__ == "__main__":
    # statistic(OUT)
    preprocess_split_data(DATASET, OUT)
    #build_word2vec()
    # load_vectors()
    #preprocess_split_data(DATASET, OUT)
    # print(text_normalize(example_content))