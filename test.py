# -*- coding: utf-8 -*-

import os
import shutil

src_bot_path = os.path.join("bot_resources", "bot1")
bot_path = os.path.join("bot_resources", "bot2")
shutil.copytree(src_bot_path, bot_path)
