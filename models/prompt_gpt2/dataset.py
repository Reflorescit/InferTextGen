
from torch.utils.data import Dataset
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import pdb
from tqdm import tqdm

import inflect
inflection_engine = inflect.engine()

from typing import List, Dict, Callable
from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

import sys
sys.path.append('../')
sys.path.append('../..')
from gpt2_zeroshot.gpt2_zeroshot import manual_prompt

class PromptDatasetProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None
    
    def article(self, word):
        return "an" if word[0] in ['a', 'e', 'i', 'o', 'u'] else "a"

    def vp_present_participle(self, phrase):
        doc = nlp(phrase)
        return ' '.join([
            inflection_engine.present_participle(token.text) if token.pos_ == "VERB" and token.tag_ != "VGG" else token.text
            for token in doc
        ])

    def posessive(self, word):
        if inflection_engine.singular_noun(word) is False:
            return "have"
        else:
            return "has"
    
    def get_examples(self, data_path: str, analysis=False, skip_none=True) -> List[InputExample]:
        data = pd.read_csv(data_path, sep='\t').dropna()
        examples = []
        if skip_none:
            data = data[data['tail_event']!='none']
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            text_a = row['head_event'].strip()
            text_b = row['relation']
            tgt_text = row['tail_event'].strip()
            if analysis and 'prompt' not in data.columns:
                meta = {'prompt': manual_prompt(kg='atomic2020', head=text_a, relation=text_b, tail=tgt_text)}
            elif 'prompt' in data.columns:
                meta = {'prompt': row['prompt']}
            else:
                meta= {}
            examples.append(InputExample(guid=idx, text_a=text_a, text_b=text_b, tgt_text=tgt_text, meta=meta))
        print("len of dataset: ", len(data))
        return examples

