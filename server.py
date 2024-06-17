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
import logging

from streaming_server import StreamingServer


def main():
    port = 6006
    # Number of threads for NN computation and decoding.
    nn_pool_size = 1
    # Max batch size for computation. Note if there are not enough
    # requests in the queue, it will wait for max_wait_ms time. After that,
    # even if there are not enough requests, it still sends the
    # available requests in the queue for computation.
    max_batch_size = 3
    # Max time in millisecond to wait to build batches for inference.
    # If there are not enough requests in the stream queue to build a batch
    # of max_batch_size, it waits up to this time before fetching available
    # requests for computation.
    max_wait_ms = 10
    # Max message size in bytes.
    # The max size per message cannot exceed this limit.
    max_message_size = 1 << 20
    # Max number of messages in the queue for each connection.
    max_queue_size = 32
    # Maximum number of active connections. The server will refuse
    # to accept new connections once the current number of active connections
    # equals to this limit.
    max_active_connections = 200

    server = StreamingServer(
        nn_pool_size=nn_pool_size,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_message_size=max_message_size,
        max_queue_size=max_queue_size,
        max_active_connections=max_active_connections,
    )
    asyncio.run(server.run(port))


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
