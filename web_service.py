# -*- coding: utf-8 -*-

import json

import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, jsonify
from flask import request
from gevent import pywsgi
from sklearn.metrics.pairwise import cosine_similarity

from config import *

app = Flask(__name__)


@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    """
    数据\t分隔，第0列为label，1列为content
    input json:
    {
        "bot_name": "xxxxxx",  # 要查询的bot name
        "operate": "upsert",  # 操作。upsert：更新或新增；delete：删除
    }

    return:
    {
        'code': 0,
        'msg': 'success'
    }
    """
    start = datetime.datetime.now()

    resq_data = json.loads(request.get_data())
    bot_name = resq_data["bot_name"].strip()
    operate = resq_data["operate"].strip()

    if operate == "upsert":
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
        del bot_sent_label_vec[bot_name]
        del bot_predict_fn[bot_name]
        if res_data == {}:
            result = {'code': 1, 'msg': 'train failed', 'time_cost': time_cost(start)}
        else:
            result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start), 'data': res_data}
        return jsonify(result)
    elif operate == "delete":
        # 删除bot
        try:
            shutil.rmtree(os.path.join(BOT_SRC_DIR, bot_name))
            del bot_sent_label_vec[bot_name]
            del bot_predict_fn[bot_name]
        except:
            print(bot_name, "deleted already...")
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
    else:
        return {'code': -1, 'msg': 'unsupported operation', 'time_cost': time_cost(start)}


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    input json:
    {
        "bot_name": "xxxxxx",  # 要查询的bot name
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
    bot_name = resq_data["bot_name"].strip()

    # 模型预测
    if bot_name not in bot_predict_fn:
        export_dir = os.path.join(BOT_SRC_DIR, bot_name, "export")
        bot_predict_fn[bot_name] = tf.contrib.predictor.from_saved_model(get_export_dir(export_dir))
        print(bot_name, "model finished reloading...")

        train_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/train.txt")
        lines = read_file(train_file_path)
        labels = [line_.split("\t")[0] for line_ in lines]
        sents = [line_.split("\t")[1] for line_ in lines]
        bert_sent_vecs = get_bert_sent_vecs(bot_predict_fn[bot_name], sents)
        bot_sent_label_vec[bot_name] = np.array([[sents[i], labels[i], bert_sent_vecs[i]] for i in range(len(sents))])

    sent_label_vecs = bot_sent_label_vec[bot_name]
    predict_fn = bot_predict_fn[bot_name]

    query_output = get_bert_sent_vecs(predict_fn, [query])[0]
    base_vecs = sent_label_vecs[:, 2]
    merge_vecs = [query_output] + list(base_vecs)

    cos_simi_m = cosine_similarity(merge_vecs)
    cos_simis = list(cos_simi_m[0])[1:]
    score = max(cos_simis)
    max_idx = cos_simis.index(score)
    label = sent_label_vecs[:, 1][max_idx]
    same_sent = sent_label_vecs[:, 0][max_idx]

    bot_last_use[bot_name] = datetime.datetime.now()
    result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start),
              'data': {'label': label, 'score': score, 'same_sent': same_sent}}
    return jsonify(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
