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

from .grpcapi import rephraser_pb2, rephraser_pb2_grpc
from .model import Model


class Servicer(rephraser_pb2_grpc.RephraserServicer):
    def __init__(self, model: Model):
        super().__init__()
        self._model: Model = model

    def Rephrase(self, request, context):
        sequences = self._model.rephrase(
            text=request.text,
            temperature=request.temperature,
            sample=request.sample,
            num_sequences=request.num_sequences,
        )

        return rephraser_pb2.RephraseReply(
            sequences=[
                rephraser_pb2.Sequence(
                    text=seq.text,
                    score=seq.score,
                )
                for seq in sequences
            ]
        )
