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
        <td>64.34%</td>
        <td>63.89%</td>
    </tr>
    <tr>
        <td>TextRNN</td>
        <td>69.35%</td>
        <td>68.98%</td>
    </tr>
    <tr>
        <td>TextCNN [2]</td>
        <td>68.08%</td>
        <td>67.72%</td>
    </tr>
    <tr>
        <td>RCNN [3]</td>
        <td>68.00%</td>
        <td>67.72%</td>
    </tr>
    <tr>
        <td>LSTM + Attention [4]</td>
        <td>67.27%</td>
        <td>66.70%</td>
    </tr>
     <tr>
        <td>HAN</td>
        <td>__.__%</td>
        <td>__.__%</td>
    </tr>
</table>

## Model Releases
- [x] <b>FastText</b> released with the paper [Bag of tricks for efficient text classification](https://arxiv.org/abs/1607.01759) by Joulin, Armand, et al.
- [x] <b>TextRNN</b>
- [x] <b>TextCNN</b> released with the paper [Convolutional neural networks for sentence classification](https://arxiv.org/abs/1408.5882) by Kim, Yoon.
- [x] <b>RCNN</b> released with the paper [Recurrent convolutional neural networks for text classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf) by Lai, Siwei, et al.
- [x] <b>LSTM + Attention</b> released with the paper [Text classification research with attention-based recurrent neural networks](https://pdfs.semanticscholar.org/7ac1/e870f767b7d51978e5096c98699f764932ca.pdf) by Du, Changshun, and Lei Huang.
- [ ] <b>Transformer</b>
- [ ] <b>BERT</b>
- [ ] <b>Hierarchical Attention Network</b>
- [ ] <b>Dynamic Memory Network</b>

## References
[1] Joulin, Armand, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. "Bag of Tricks for Efficient Text Classification." EACL 2017 (2017): 427.

[2] Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.

[3] Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." In Proc. Conference of the Association for the Advancement of Artificial Intelligence (AAAI). 2015.

[4] Du, Changshun, and Lei Huang. "Text classification research with attention-based recurrent neural networks." International Journal of Computers Communications & Control 13.1 (2018): 50-61.