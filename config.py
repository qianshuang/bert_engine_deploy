# -*- coding: utf-8 -*-

import schedule
import threading
import time

from bert_common import *

bot_last_use = {}
bot_need_retrain = {}


# 每天把超过7天未使用bot模型从内存卸载
def run_resources():
    for _bot_name_ in os.listdir(BOT_SRC_DIR):
        # 卸载最近未使用bot
        if _bot_name_ in bot_last_use and time_cost(bot_last_use[_bot_name_],
                                                    "day") > 7 and _bot_name_ in bot_predict_fn:
            del bot_sent_label_vec[_bot_name_]
            del bot_predict_fn[_bot_name_]
            print(_bot_name_, 'long time no use, unloaded...')

        # 只保留最近的两个export
        _export_dir_ = os.path.join(BOT_SRC_DIR, _bot_name_, "export")
        _dirs_ = os.listdir(_export_dir_)
        _dirs_.sort(reverse=True)
        for d in _dirs_[1:]:
            shutil.rmtree(os.path.join(_export_dir_, d))

    # 重训练bot
    for i in sorted(bot_need_retrain.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        try:
            train_from_scratch(i[0])
        except:
            print("too many training tasks, try tomorrow ---", bot_need_retrain)
            break
        load_model_vec(i[0])
        bot_need_retrain.pop(i[0])


schedule.every().day.do(run_resources)


# 多线程调度
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)


threading.Thread(target=run_schedule).start()
