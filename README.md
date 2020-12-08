# textclassifiers
Textclassifiers: Collection of Text Classification/Document Classification/Sentence Classification/Sentiment Analysis models for PyTorch

Install dependencies:

`pip3 install -r requirements.txt`

## Run the code
### Train
`python3 run.py --mode train --config configs/fasttext_config.yaml`

### Query Well formedness result
The overall model performances on test set. 

**Note: The test's model parameter configuration is saved in `./examples/ag_news`

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
        <td>66.33%</td>
        <td>66.20%</td>
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
        <td>Transformer [5]</td>
        <td>68.31%</td>
        <td>67.78%</td>
    </tr>
    <tr>
        <td>BERT [6]</td>
        <td>__.__%</td>
        <td>__.__%</td>
    </tr>
     <tr>
        <td>HAN [7]</td>
        <td>__.__%</td>
        <td>__.__%</td>
    </tr>
    <tr>
        <td>DNN</td>
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
- [x] <b>Transformer</b> released with the paper [Attention is all you need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf) by Vaswani, Ashish, et al.
- [ ] <b>BERT</b> released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) by Devlin, Jacob, et al.
- [ ] <b>Hierarchical Attention Network</b> released with the paper [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174.pdf) by Yang, Zichao, et al.
- [ ] <b>Dynamic Memory Network</b>

## References
[1] Joulin, Armand, Edouard Grave, and Piotr Bojanowski Tomas Mikolov. "Bag of Tricks for Efficient Text Classification." EACL 2017 (2017): 427.

[2] Kim, Yoon. "Convolutional Neural Networks for Sentence Classification." Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2014.

[3] Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." In Proc. Conference of the Association for the Advancement of Artificial Intelligence (AAAI). 2015.

[4] Du, Changshun, and Lei Huang. "Text classification research with attention-based recurrent neural networks." International Journal of Computers Communications & Control 13.1 (2018): 50-61.

[5] Vaswani, Ashish, et al. "Attention is all you need." Proceedings of the 31st International Conference on Neural Information Processing Systems. Curran Associates Inc., 2017.

[6] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.

[7] Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.