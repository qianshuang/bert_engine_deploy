# -*- coding: utf-8 -*-

import datetime

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


def time_cost(start, type_="sec"):
    interval = datetime.datetime.now() - start
    if type_ == "sec":
        return interval.total_seconds()
    elif type_ == "day":
        return interval.days


def read_bert_eval_res_to_dict(bert_eval_file):
    eval_res = {}
    for line in read_file(bert_eval_file):
        kv = line.split("=")
        eval_res[kv[0].strip()] = float(kv[1].strip())
    return eval_res
