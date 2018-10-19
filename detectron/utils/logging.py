# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
from email.mime.text import MIMEText
import json
import logging
import numpy as np
import smtplib
import sys, os

from detectron.core.config import cfg
from detectron.core.config import get_output_dir

# Print lower precision floating point values than default FLOAT_REPR
json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')


class StatsLogger(object):
    def __init__(self, save=True, sort_keys=True):
        self.save = save
        if save:
            self.output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
            self.log_path = os.path.join(self.output_dir, 'log.json')
            open(self.log_path, "w").close()
        self.sort_keys = sort_keys

    def log_json(self, stats_main, stats_extra):
        json_stats_extra = json.dumps(stats_extra, sort_keys=self.sort_keys)
        stats_main.update(stats_extra)
        json_stats = json.dumps(stats_main)
        print("-" * 100)
        print("iter: {}, loss: {:.4}, eta: {}, time: {:.0f} ms, lr: {:.3}".format(
            stats_main['iter'],
            stats_main['loss'],
            stats_main['eta'],
            stats_main['time'] * 1000,
            stats_main['lr'],
        ))
        stats_str_list = ["{}: {:.3}".format(k, v) for k, v in stats_extra.items()]
        stats_str_list.sort()
        print(", ".join(stats_str_list))
        if self.save:
            with open(self.log_path, "a") as f:
                f.write(json_stats + "\n")



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count


def send_email(subject, body, to):
    s = smtplib.SMTP('localhost')
    mime = MIMEText(body)
    mime['Subject'] = subject
    mime['To'] = to
    s.sendmail('detectron', to, mime.as_string())


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger
