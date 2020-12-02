# textclassifiers
Textclassifiers: Collection of Text Classification models for PyTorch

Install dependencies:

`pip3 install -r requirements.txt`

## Run the code
### Train
`python3 train.py --config configs/fasttext_config.yaml`

### Query Well formedness result
| Models              |       Score             ||
|---------------------|------------|------------|
|                     | Accuracy   | F1 Score   | 
| FastText            |  61,38%    |  61,32%    | 
| TextRNN             |  68,29%    |  68,08%    |
| TextCNN             |  ..,..%    |  ..,..%    |

## Model Releases
- [x] FastText
- [x] TextRNN
- [ ] TextCNN
- [ ] RCNN
- [ ] Seq2Seq Attention
- [ ] Transformer
- [ ] BERT
- [ ] Hierarchical Attention Network
- [ ] Dynamic Memory Network