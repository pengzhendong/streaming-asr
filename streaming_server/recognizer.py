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

from modelscope import snapshot_download
from sherpa_onnx import OnlineRecognizer
from wetext import Normalizer

from .punctuation import Punctuation


class Recognizer(OnlineRecognizer):
    def __init__(
        self,
        model_id="pengzhendong/streaming-paraformer-zh-en",
        sample_rate=16000,
        add_punctuation=True,
        normalize=True,
    ):
        repo_dir = snapshot_download(model_id)
        self.recognizer = (
            super()
            .from_paraformer(
                tokens=f"{repo_dir}/tokens.txt",
                encoder=f"{repo_dir}/encoder.onnx",
                decoder=f"{repo_dir}/decoder.onnx",
                num_threads=1,
                provider="cpu",
                sample_rate=sample_rate,
                decoding_method="greedy_search",
            )
            .recognizer
        )

        self.punct = None
        if add_punctuation:
            self.punct = Punctuation()

        self.normalizer = None
        if normalize:
            self.normalizer = Normalizer(lang="zh", operator="itn")

    def post_process(self, text):
        if self.punct is not None:
            text = self.punct.add_punctuation(text)
        if self.normalizer is not None:
            text = self.normalizer.normalize(text)
        return text
