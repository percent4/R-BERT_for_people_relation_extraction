# R-BERT for People Relation Classification

本项目采用`R-BERT模型`: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)对人物关系进行分类，提升效果明显，在测试集上的F1值达到85%。

## 数据集

- 共3901条标注样本，训练集：测试集=8:2
- 标注样本：`亲戚  1837年6月20日，<e1>威廉四世</e1>辞世，他的侄女<e2>维多利亚</e2>即位。`，其中`亲戚`为关系，`威廉四世`为实体1（entity_1），`维多利亚`为实体2（entity_2）。
- 每一种关系的标注数量如下图:

<p float="left" align="center">
    <img width="600" src="" />  
</p>

## 模型结构

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

1. **Get three vectors from BERT.**
   - [CLS] token vector
   - averaged entity_1 vector
   - averaged entity_2 vector
2. **Pass each vector to the fully-connected layers.**
   - dropout -> tanh -> fc-layer
3. **Concatenate three vectors.**
4. **Pass the concatenated vector to fully-connect layer.**
   - dropout -> fc-layer

- **_Exactly the SAME conditions_** as written in paper.
  - **Averaging** on `entity_1` and `entity_2` hidden state vectors, respectively. (including \$, # tokens)
  - **Dropout** and **Tanh** before Fully-connected layer.
  - **No [SEP] token** at the end of sequence. (If you want add [SEP] token, give `--add_sep_token` option)

## 运行环境

- python>=3.6
- 第三方模块参考: `requiments.txt`

## 模型训练

```bash
$ python3 main.py --do_train --do_eval
```

模型预测文件位于`eval`目录下的`proposed_answers.txt`。

## 模型评估

```bash
$ python3 evalaute.py
# weighted avgage F1 = 85.35%
```

真实关系文件位于`eval`目录下的`true_answers.txt`，模型预测文件位于`eval`目录下的`proposed_answers.txt`。

## 模型预测

```bash
$ python3 predict.py
```
示例的模型预测输入和输出可以参考`sample_pred_in.txt`和`sample_pred_in.txt`。

## References

- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)
- [R-BERT](https://github.com/monologg/R-BERT)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/pdf/1905.08284.pdf)
