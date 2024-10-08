# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import numpy as np

from funasr import AutoModel


class Emotion2Vec:
    def __init__(self, model_id="iic/emotion2vec_base_finetuned"):
        self.model = AutoModel(model=model_id)

    def inference(self, samples, granularity="utterance"):
        assert granularity in ["frame", "utterance"]
        res = self.model.generate(
            samples,
            granularity=granularity,
            extract_embedding=False,
        )[0]
        return res["labels"][np.argmax(res["scores"])]
