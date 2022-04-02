# Copyright 2022 Matteo Grella, Marco Nicola
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
from concurrent import futures

import grpc
from .servicer import Servicer
from .model import Model
from .grpcapi import rephraser_pb2_grpc

DEFAULT_MODEL_NAME = 'geckos/pegasus-fined-tuned-on-paraphrase'
DEFAULT_MODELS_PATH = 'models'


def main() -> None:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level='INFO')

    parser = argparse.ArgumentParser(description='Run the rephrasing server.')
    parser.add_argument('-m', '--model', dest='model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('-p', '--path', dest='path', default=DEFAULT_MODELS_PATH)
    parser.add_argument('-w', '--max-workers', dest='max_workers', type=int, default=4)
    parser.add_argument('-a', '--address', dest='address', default='0.0.0.0:8080')
    args = parser.parse_args()

    model = Model(args.model, args.path)
    servicer = Servicer(model)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))

    rephraser_pb2_grpc.add_RephraserServicer_to_server(servicer, server)
    server.add_insecure_port(args.address)
    logging.info(f'serving on {args.address}')
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt')

    logging.info('Bye!')


if __name__ == '__main__':
    main()
