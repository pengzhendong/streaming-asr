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
import json
import logging

import sounddevice as sd
import wave
import websockets


async def send_wav(ws):
    try:
        wf = wave.open("test_16k.wav")
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        chunk_ms = 10 # 10ms each chunk
        chunk_frames = chunk_ms * framerate // 1000
        import numpy as np
        for _ in range(0, n_frames, chunk_frames):
            frames = wf.readframes(chunk_frames)
            await ws.send(frames)
            # Simulate streaming
            await asyncio.sleep(chunk_ms / 1000)
        await ws.send("Done")
    except Exception as e:
        logging.info(e)


async def send(ws):
    try:
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)
        print(devices)
        default_input_device_idx = sd.default.device[0]
        print(f'Use default device: {devices[default_input_device_idx]["name"]}')

        sample_rate = 16000
        samples_per_read = int(0.1 * sample_rate)
        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while True:
                samples, _ = s.read(samples_per_read)
                samples = (samples[:, 0] * 32768).astype("int16")
                await ws.send(samples.data)
                await asyncio.sleep(0.1)  # 让出 CPU 资源给其他协程
        await ws.send("Done")
    except Exception as e:
        logging.info(e)


async def receive(ws):
    try:
        while True:
            message = await ws.recv()
            logging.info(json.loads(message))
    except Exception as e:
        logging.info(e)


async def main():
    ws = await websockets.connect("ws://127.0.0.1:6006")
    send_task = asyncio.create_task(send(ws))
    # send_task = asyncio.create_task(send_wav(ws))
    receive_task = asyncio.create_task(receive(ws))
    try:
        await asyncio.gather(send_task, receive_task)
    except Exception as e:
        logging.info(e)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    asyncio.run(main())
