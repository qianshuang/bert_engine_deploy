# -*- coding: utf-8 -*-
import datetime
import json

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
        "operate": "update",  # 操作。update：更新bot；delete：删除bot；new：新增bot；copy：复制bot
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

    if operate == "new":
        try:
            res_data = train_from_scratch(bot_name)
            load_model_vec(bot_name)
        except:
            return {'code': 2, 'msg': 'another training task is processing, please try again later',
                    'time_cost': time_cost(start)}
        if res_data == {}:
            result = {'code': 1, 'msg': 'train failed', 'time_cost': time_cost(start)}
        else:
            result = {'code': 0, 'msg': 'success', 'time_cost': time_cost(start), 'data': res_data}
        return jsonify(result)
    elif operate == "copy":
        # 复制bot。一方面不用从头训练，直接复用原始bot的能力；另一方面避免误删除bot
        src_bot_name = resq_data["src_bot_name"].strip()
        src_bot_path = os.path.join(BOT_SRC_DIR, src_bot_name)
        bot_path = os.path.join(BOT_SRC_DIR, bot_name)
        shutil.copytree(src_bot_path, bot_path)
        load_model_vec(bot_name)
        if src_bot_name in bot_need_retrain:
            bot_need_retrain[bot_name] = datetime.datetime.now()
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
    elif operate == "update":
        # 增量计算向量
        if bot_name not in bot_predict_fn:
            load_model_vec(bot_name)
        sent_label = {}
        sent_vec = {}
        for item in bot_sent_label_vec[bot_name]:
            sent_label[item[0]] = item[1]
            sent_vec[item[0]] = item[2]
        data_file_path = os.path.join(BOT_SRC_DIR, bot_name, "data/data.txt")
        lines = read_file(data_file_path)
        labels = [line_.split("\t")[0] for line_ in lines]
        sents = [line_.split("\t")[1] for line_ in lines]
        sent_label_dict = dict(zip(sents, labels))

        need_compute = list(set(sents) - sent_label.keys())
        query_outputs = get_bert_sent_vecs(bot_predict_fn[bot_name], need_compute)
        sent_vec_dict = dict(zip(need_compute, query_outputs))

        final_sent_label_vecs = []
        for sent in sents:
            if sent in need_compute:
                final_sent_label_vecs.append([sent, sent_label_dict[sent], sent_vec_dict[sent]])
            else:
                final_sent_label_vecs.append([sent, sent_label[sent], sent_vec[sent]])
        bot_sent_label_vec[bot_name] = np.array(final_sent_label_vecs)
        bot_need_retrain[bot_name] = datetime.datetime.now()
        return {'code': 0, 'msg': 'success', 'time_cost': time_cost(start)}
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
        load_model_vec(bot_name)

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
              'data': {'label': label, 'score': float(score), 'same_sent': same_sent}}
    return jsonify(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8088), app)
    server.serve_forever()
