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
        <td>FastText [1]</td>
        <td>61.38%</td>
        <td>61.32%</td>
    </tr>
    <tr>
        <td>TextRNN</td>
        <td>68.29%</td>
        <td>68.08%</td>
    </tr>
    <tr>
        <td>TextCNN [2]</td>
        <td>68.05%</td>
        <td>67.35%</td>
    </tr>
    <tr>
        <td>RCNN [3]</td>
        <td>__.__%</td>
        <td>__.__%</td>
    </tr>
</table>

## Model Releases
- [x] <b>FastText</b> released with the paper [Bag of tricks for efficient text classification](https://arxiv.org/abs/1607.01759) by Joulin, Armand, et al.
- [x] <b>TextRNN</b>
- [x] <b>TextCNN</b> released with the paper [Convolutional neural networks for sentence classification](https://arxiv.org/abs/1408.5882) by Kim, Yoon.
- [ ] <b>RCNN</b> released with the paper [Recurrent convolutional neural networks for text classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf) by Lai, Siwei, et al.
- [ ] <b>Seq2Seq Attention</b>
- [ ] <b>Transformer</b>
- [ ] <b>BERT</b>
- [ ] <b>Hierarchical Attention Network</b>
- [ ] <b>Dynamic Memory Network</b>

## References
[1] Joulin, Armand, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. "Bag of Tricks for Efficient Text Classification." EACL 2017 (2017): 427.

[2] Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.

[3] Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." In Proc. Conference of the Association for the Advancement of Artificial Intelligence (AAAI). 2015.