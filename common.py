# -*- coding: utf-8 -*-

import datetime
import numpy as np

BOT_SRC_DIR = "bot_resources"


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    return [line.strip() for line in open(filename).readlines() if line.strip() != ""]


def write_file(filename, content):
    open_file(filename, mode="w").write(content)


def append_file(filename, content):
    open_file(filename, mode="a").write(content)


def write_lines(filename, list_res):
    test_w = open_file(filename, mode="w")
    for j in list_res:
        test_w.write(j + "\n")


def time_cost(start):
    end_time = datetime.datetime.now()
    return str(end_time - start).split('.')[0]


def get_label_score_by_probs(probabilities, all_labels):
    max_idx = np.argmax(probabilities)
    return all_labels[max_idx], float(probabilities[max_idx])


def read_bert_eval_res_to_dict(bert_eval_file):
    eval_res = {}
    for line in read_file(bert_eval_file):
        kv = line.split("=")
        eval_res[kv[0].strip()] = float(kv[1].strip())
    return eval_res
