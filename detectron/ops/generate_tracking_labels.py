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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from detectron.utils import blob as blob_utils


class GenerateTrackingLabelsOp(object):

    def forward(self, inputs, outputs):
        """See modeling.detector.GenerateTrackingLabels for inputs/outputs
        documentation.
        """

        track_ids_one = inputs[0].data
        track_ids_two = inputs[1].data

        tracking_labels = np.array([id_one == id_two for id_one in track_ids_one for id_two in track_ids_two], dtype=np.int32)
        n_matches = sum(tracking_labels)
        
        assert n_matches > 0, "Image pair with no matches encountered"
        assert len(tracking_labels) - n_matches > 0, "Image pair with only matches encountered"

        blob_utils.py_op_copy_blob(tracking_labels, outputs[0])
