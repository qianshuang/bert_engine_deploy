# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import tensorflow as tf
import numpy as np
import pandas as pd

import shutil

from bert import tokenization
from bert import modeling

from common import *

max_seq_length = 128
vocab_file = "bert/uncased_L-12_H-768_A-12/vocab.txt"
bert_config_file = "bert/uncased_L-12_H-768_A-12/bert_config.json"
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

bot_predict_fn = {}
bot_sent_label_vec = {}


def build_bert_cmd(bot_name, all_labels, num_train_epochs):
    bot_dir = str(os.path.join(BOT_SRC_DIR, bot_name))
    cmd = ["python3", "bert/run_classifier.py",
           "--data_dir=" + bot_dir + "/data",
           "--bert_config_file=" + bert_config_file,
           "--task_name=comm100",
           "--vocab_file=" + vocab_file,
           "--output_dir=" + bot_dir + "/output",
           "--init_checkpoint=bert/uncased_L-12_H-768_A-12/bert_model.ckpt",
           "--do_train=True",
           "--do_eval=True",
           "--num_train_epochs=" + str(num_train_epochs),
           "--all_labels=" + all_labels]
    return cmd


def train(bot_name, all_labels):
    former_eval_accuracy = 0
    bert_eval_file = os.path.join(BOT_SRC_DIR, bot_name, "output/eval_results.txt")

    for epoch in range(1, sys.maxsize):
        cmd = build_bert_cmd(bot_name, all_labels, epoch)
        ret_res = subprocess.call(cmd)
        if ret_res != 0:
            return {}

        eval_res = read_bert_eval_res_to_dict(bert_eval_file)
        eval_res["epoch"] = epoch

        eval_accuracy = eval_res["eval_accuracy"]
        if abs(eval_accuracy - former_eval_accuracy) < 0.01:
            return eval_res
        else:
            former_eval_accuracy = eval_accuracy


def convert_single_example(text):
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokens_a = tokenizer.tokenize(text)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return input_ids, input_mask, segment_ids


def get_export_dir(export_dir):
    dirs = os.listdir(export_dir)
    dirs.sort(reverse=True)
    return os.path.join(export_dir, dirs[0])


def get_bert_sent_vecs(bot_predict_fn_, sent_list):
    if len(sent_list) == 0:
        return np.array([])

    feed_dict = {"input_ids": [], "input_mask": [], "segment_ids": []}
    for line in sent_list:
        input_ids, input_mask, segment_ids = convert_single_example(line.strip())
        feed_dict["input_ids"].append(input_ids)
        feed_dict["input_mask"].append(input_mask)
        feed_dict["segment_ids"].append(segment_ids)
    prediction = bot_predict_fn_(feed_dict)
    query_output = prediction["query_output"]
    return query_output


def load_model_vec(bot_name):
    export_dir = os.path.join(BOT_SRC_DIR, bot_name, "export")
    bot_predict_fn[bot_name] = tf.contrib.predictor.from_saved_model(get_export_dir(export_dir))
    print(bot_name, "model finished reloading...")

    train_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/train.txt")
    lines = read_file(train_file_path)
    labels = [line_.split("\t")[0] for line_ in lines]
    sents = [line_.split("\t")[1] for line_ in lines]
    bert_sent_vecs = get_bert_sent_vecs(bot_predict_fn[bot_name], sents)
    bot_sent_label_vec[bot_name] = np.array([[sents[i], labels[i], bert_sent_vecs[i]] for i in range(len(sents))])


def train_from_scratch(bot_name):
    data_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/data.txt")
    train_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/train.txt")
    test_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/test.txt")
    label_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/label.txt")
    export_dir = os.path.join(BOT_SRC_DIR, bot_name, "export")

    # 拆分训练测试集
    write_lines(data_file_path, read_file(data_file_path))
    df = pd.read_csv(data_file_path, sep='\t', header=None)
    df_test = df.sample(frac=0.1, axis=0)
    df_train = df[~df.index.isin(df_test.index)]
    df_test.to_csv(test_file_path, sep='\t', index=False, header=False)
    df_train.to_csv(train_file_path, sep='\t', index=False, header=False)

    # 自动化训练 + 模型导出 + 模型加载
    all_labels = list(set(df[0].values))
    write_lines(label_file_path, all_labels)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    res_data = train(bot_name, ",".join(all_labels))
    return res_data
