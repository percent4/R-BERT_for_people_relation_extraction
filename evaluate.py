# -*- coding: utf-8 -*-
# @Time : 2021/3/16 13:52
# @Author : Jclian91
# @File : evaluate.py
# @Place : Yangpu, Shanghai
from sklearn.metrics import classification_report


# get relation
def get_label(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        relations = [_.strip().split("\t")[-1] for _ in f.readlines()]
    return relations


# model evaluate
if __name__ == '__main__':
    true_relations = get_label("./eval/true_answer.txt")
    predict_relations = get_label("./eval/proposed_answers.txt")
    result = classification_report(y_true=true_relations, y_pred=predict_relations, digits=4)
    print(result)