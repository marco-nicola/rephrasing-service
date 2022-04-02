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

import logging
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODELS_PATH = "models"
MODEL_NAME = 'geckos/pegasus-fined-tuned-on-paraphrase'


@dataclass
class Sequence:
    text: str
    score: float


class Model:
    def __init__(self, name: str, cache_dir: str):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logging.info(f'Loading model...')
        self._model = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir)
        self._model.to(self._device)

        logging.info(f'Loading tokenizer...')
        self._tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

        logging.info(f'Loading done.')

    def rephrase(self, text: str, temperature: float, sample: bool, num_sequences: int) -> list[Sequence]:
        text += ' </s>'
        encoding = self._tokenizer.encode_plus(text, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].to(self._device)
        attention_masks = encoding['attention_mask'].to(self._device)
        beam_outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=sample,
            max_length=256,
            temperature=temperature,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=num_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )
        return [
            Sequence(
                text=self._tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                score=float(score),
            )
            for (sequence, score) in zip(beam_outputs.sequences, beam_outputs.sequences_scores)
        ]
