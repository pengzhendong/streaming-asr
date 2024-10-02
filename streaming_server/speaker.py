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
import sherpa_onnx

from modelscope import model_file_download


wespeaker = {}


class WeSpeaker:
    def __init__(self, model_name, sample_rate=16000):
        model = model_file_download("pengzhendong/speaker-identification", model_name)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model, num_threads=1, provider="cpu"
        )
        if not config.validate():
            raise ValueError(f"Invalid config. {config}")

        self.sample_rate = sample_rate
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def compute(self, samples):
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=samples)
        stream.input_finished()
        if self.extractor.is_ready(stream):
            return np.array(self.extractor.compute(stream))
        else:
            return None


class Speaker:
    def __init__(self, threshold=0.5, max_num_speakers=3, sample_rate=16000):
        if sample_rate not in wespeaker:
            model_name = "wespeaker_zh_cnceleb_resnet34_LM.onnx"
            wespeaker[sample_rate] = WeSpeaker(model_name, sample_rate)
        self.model = wespeaker[sample_rate]
        self.threshold = threshold
        self.max_num_speakers = max_num_speakers
        self.embeddings = []

    @staticmethod
    def similarities(embedding, embeddings):
        return np.dot(embedding, np.array(embeddings).T) / (
            np.linalg.norm(embedding) * np.linalg.norm(embeddings, axis=1)
        )

    def get_speaker_id(self, samples):
        embedding = self.model.compute(samples)
        if embedding is None:
            return -1
        if len(self.embeddings) == 0:
            self.embeddings.append(embedding)
            return 0
        similarities = self.similarities(embedding, self.embeddings)
        idx = int(np.argmax(similarities))
        if similarities[idx] < self.threshold:
            idx = len(self.embeddings)
            self.embeddings.append(embedding)
        return idx
