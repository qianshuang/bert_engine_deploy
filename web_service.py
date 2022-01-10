# -*- coding: utf-8 -*-

import json
import shutil
import pandas as pd
import tensorflow as tf

from flask import Flask, jsonify
from flask import request
from gevent import pywsgi

from config import *

app = Flask(__name__)


@app.route('/init_train', methods=['GET', 'POST'])
def init_train():
    """
    数据\t分隔，第0列为label，1列为content
    input json:
    {
        "bot_name": "xxxxxx",  # 要查询的bot name
    }

    return:
    {
        'code': 0,
        'msg': 'success',
        'time_cost': 30
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    bot_name = resq_data["bot_name"].strip()
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
    if res_data == {}:
        result = {'code': -1, 'msg': 'bert train failed!', 'time_cost': time_cost(start)}
    else:
        bot_predict_fn[bot_name] = tf.contrib.predictor.from_saved_model(get_export_dir(export_dir))
        # 加载向量
        lines = read_file(train_file_path)
        labels = [line_.split("\t")[0] for line_ in lines]
        sents = [line_.split("\t")[1] for line_ in lines]
        bert_sent_vecs = get_bert_sent_vecs(sents)

        sent_label_vecs = []
        for i in range(len(sents)):
            sent_label_vecs.append({sents[i]: [labels[i], bert_sent_vecs[i]]})
        bot_sent_label_vec[bot_name] = sent_label_vecs

        result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start), 'data': res_data}
    return jsonify(result)


@app.route('/incre_train', methods=['GET', 'POST'])
def incre_train():
    """
    input json:
    {
        "data": "xxx"  # 训练数据，数据\t分隔，第0列为label，1列为content
    }

    return:
    {
        'code': 0,
        'msg': 'success',
        'time_cost': 30
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    data = resq_data["data"].strip()
    append_file(TRAIN_FILE, data + "\n")

    # 自动化训练 + 模型导出 + 模型加载
    all_labels = list(set([line.strip().split("\t")[0].strip() for line in read_file(TRAIN_FILE)]))
    write_lines(LABEL_FILE, all_labels)
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)

    bert_eval_dict = read_bert_eval_res_to_dict(bert_eval_file)
    res_data = train(",".join(all_labels), int(bert_eval_dict["epoch"]) + 1, bert_eval_dict["eval_accuracy"])
    globals()['predict_fn'] = tf.contrib.predictor.from_saved_model(get_export_dir())

    result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start), 'data': res_data}
    return jsonify(result)


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    input json:
    {
        "query": "xxx"  # 用户query
    }

    return:
    {
        'code': 0,
        'msg': 'success',
        'time_cost': 30,
        'data': {'label': 'xxx', 'score': 0.998}
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    query = resq_data["query"].strip()

    # 模型预测
    query_output = get_bert_sent_vecs([query])[0]

    result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start), 'data': {'label': label, 'score': score}}
    return jsonify(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
