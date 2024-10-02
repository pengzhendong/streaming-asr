# Copyright     2022-2023  Xiaomi Corp.
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

import asyncio
import http
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import sherpa_onnx
import websockets
from pysilero import VADIterator

from .emotion2vec import Emotion2Vec
from .recognizer import Recognizer
from .speaker import Speaker


class StreamingServer(object):
    def __init__(
        self,
        nn_pool_size: int,
        max_wait_ms: float,
        max_batch_size: int,
        max_message_size: int,
        max_queue_size: int,
        max_active_connections: int,
    ):
        """
        Args:
          nn_pool_size:
            Number of threads for the thread pool that is responsible for
            neural network computation and decoding.
          max_wait_ms:
            Max wait time in milliseconds in order to build a batch of
            `batch_size`.
          max_batch_size:
            Max batch size for inference.
          max_message_size:
            Max size in bytes per message.
          max_queue_size:
            Max number of messages in the queue for each connection.
          max_active_connections:
            Max number of active connections. Once number of active client
            equals to this limit, the server refuses to accept new connections.
          beam_search_params:
            Dictionary containing all the parameters for beam search.
        """
        self.nn_pool_size = nn_pool_size
        self.nn_pool = ThreadPoolExecutor(
            max_workers=nn_pool_size,
            thread_name_prefix="nn",
        )
        self.stream_queue = asyncio.Queue()
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        self.max_active_connections = max_active_connections
        self.current_active_connections = 0

        self.emotion2vec = Emotion2Vec()
        self.sample_rate = 16000
        self.speaker = Speaker(sample_rate=self.sample_rate)
        self.recognizer = Recognizer(sample_rate=self.sample_rate)

    async def stream_consumer_task(self):
        """This function extracts streams from the queue, batches them up, sends
        them to the neural network model for computation and decoding.
        """
        while True:
            if self.stream_queue.empty():
                await asyncio.sleep(self.max_wait_ms / 1000)
                continue

            batch = []
            try:
                while len(batch) < self.max_batch_size:
                    item = self.stream_queue.get_nowait()
                    assert self.recognizer.is_ready(item[0])
                    batch.append(item)
            except asyncio.QueueEmpty:
                pass
            stream_list = [b[0] for b in batch]
            future_list = [b[1] for b in batch]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.nn_pool,
                self.recognizer.decode_streams,
                stream_list,
            )
            for f in future_list:
                self.stream_queue.task_done()
                f.set_result(None)

    async def compute_and_decode(self, stream: sherpa_onnx.OnlineStream) -> None:
        """Put the stream into the queue and wait it to be processed by the
        consumer task.

        Args:
          stream:
            The stream to be processed. Note: It is changed in-place.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.stream_queue.put((stream, future))
        await future

    async def process_request(
        self,
        path: str,
        request_headers: websockets.Headers,
    ) -> Optional[Tuple[http.HTTPStatus, websockets.Headers, bytes]]:
        if (
            "sec-websocket-key" in request_headers
            and self.current_active_connections < self.max_active_connections
        ):
            self.current_active_connections += 1
            return None
        # Refuse new connections
        status = http.HTTPStatus.SERVICE_UNAVAILABLE  # 503
        header = {"Hint": "The server is overloaded. Please retry later."}
        response = b"The server is busy. Please retry later."
        return status, header, response

    async def handle_connection(self, socket: websockets.WebSocketServerProtocol):
        """Receive audio samples from the client, process it, and send
        decoding result back to the client.

        Args:
          socket:
            The socket for communicating with the client.
        """
        try:
            logging.info(
                "Connected: %s. Number of connections: %s/%s",
                socket.remote_address,
                self.current_active_connections,
                self.max_active_connections,
            )
            stream = self.recognizer.create_stream()
            vad_iterator = VADIterator(sample_rate=self.sample_rate, denoise=True)

            segment = 0
            last_result = ""
            while True:
                # Each message contains either a bytes buffer containing audio
                # samples in 16 kHz or contains "Done" meaning the end of utterance.
                message = await socket.recv()
                if message == "Done":
                    break
                samples = np.frombuffer(message, dtype=np.int16)
                samples = samples.astype(np.float32) / 32768
                for speech_dict, speech_samples in vad_iterator(samples):
                    stream.accept_waveform(self.sample_rate, speech_samples)
                    if "end" in speech_dict:
                        tail_padding = np.zeros(int(self.sample_rate * 0.6)).astype(
                            np.float32
                        )
                        stream.accept_waveform(self.sample_rate, tail_padding)
                    while self.recognizer.is_ready(stream):
                        await self.compute_and_decode(stream)
                        result = self.recognizer.get_result(stream)
                        if result == "" or result == last_result:
                            continue
                        last_result = result
                        await socket.send(
                            json.dumps({"text": result, "segment": segment})
                        )
                    if "end" in speech_dict and result != "":
                        speech_samples = vad_iterator.speech_samples
                        json_data = json.dumps(
                            {
                                "text": self.recognizer.post_process(result),
                                "emotion": self.emotion2vec.inference(speech_samples),
                                "speaker": self.speaker.get_speaker_id(speech_samples),
                                "segment": segment,
                                "end": True,
                            },
                            ensure_ascii=False,
                        )
                        logging.info("asr result: %s", json_data)
                        await socket.send(json_data)
                        segment += 1
                        self.recognizer.reset(stream)
            stream.input_finished()
        except websockets.exceptions.ConnectionClosedError:
            logging.info("%s disconnected", socket.remote_address)
        finally:
            # Decrement so that it can accept new connections
            self.current_active_connections -= 1
            logging.info(
                "Connected: %s. Number of connections: %s/%s",
                socket.remote_address,
                self.current_active_connections,
                self.max_active_connections,
            )

    async def run(self, port: int):
        tasks = [
            asyncio.create_task(self.stream_consumer_task())
            for _ in range(self.nn_pool_size)
        ]
        async with websockets.serve(
            self.handle_connection,
            host="0.0.0.0",
            port=port,
            max_size=self.max_message_size,
            max_queue=self.max_queue_size,
            process_request=self.process_request,
        ):
            print(
                f"Please visit one of the following addresses:\n\n  http://127.0.0.1:{port}\n"
            )
            await asyncio.Future()  # run forever
        await asyncio.gather(*tasks)  # not reachable
