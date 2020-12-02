# textclassifiers
Textclassifiers: Collection of Text Classification models for PyTorch

Install dependencies:

`pip3 install -r requirements.txt`

## Run the code
### Train
`python3 run.py --config configs/fasttext_config.yaml`

### Query Well formedness result
<table>
  <tr>
    <th rowspan="2">Model</th>
    <th align="center" colspan="2">Score</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>F1 Score </th>
  </tr>
  <tr>
    <td>FastText</td>
    <td>61.38%</td>
    <td>61.32%</td>
  </tr>
  <tr>
    <td>TextRNN</td>
    <td>68.29%</td>
    <td>68.08%</td>
  </tr>
  <tr>
    <td>TextCNN</td>
    <td>__.__%</td>
    <td>__.__%</td>
  </tr>
</table>

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