# Importing stock libraries
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from thinc.config import Config
import torch
from torch import random
from torch.nn import parallel
import torch.nn.functional as F
from torch.serialization import save
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from typing import List

from torch import nn
from tqdm.std import TRLock

# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import os for env varibles via Beaker
import os

# WandB – Import the wandb library
import wandb
import logging
import argparse

from torch import cuda
import sys
import os
import pdb
from tqdm import tqdm
sys.path.append('..')
sys.path.append('../../')
# sys.path.append('../../../')
# print(sys.path)

# from split.utils import write_items
from optparse import OptionParser

device = 'cuda' if cuda.is_available() else 'cpu'

logger = logging.getLogger("gpt2-comet")
logging.basicConfig(level=logging.INFO)

# logger.info for allenai beaker verification
logger.info(device)
logger.info(torch.cuda.device_count())

use_parallel = (torch.cuda.device_count()>1)

from mosaic.infra.modeling import train, validate, beam_generations, save_model
from mosaic.datasets.KGDataset import KGDataset
from gpt2_zeroshot.gpt2_zeroshot import manual_prompt


DEBUG = False
NUM_INST = 100
FILTER_NONE = False

RELATIONS = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires',
             'isAfter', 'HasSubEvent', 'isBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy', 
             'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant']
# EXTRA_RELS = ["gEffect", "gReact", "gWant"]

# FEW_SHOT_RELATIONS = []
FEW_SHOT_RELATIONS = ['MadeUpOf', 'NotDesires', 'xReason', 'isBefore', 'xReact', 'oWant']
# FEW_SHOT_RELATIONS = ['ObjectUse', 'CapableOf', 'AtLocation', 'xAttr']

trn_data_path = '../../atomic2020_data-feb2021/train.tsv'
dev_data_path = '../../atomic2020_data-feb2021/dev.tsv'
tst_data_path = '../../atomic2020_data-feb2021/test.tsv'

prompt_trn_path = '../../atomic2020_data-feb2021/prompt_datasets/train.tsv'
prompt_dev_path = '../../atomic2020_data-feb2021/prompt_datasets/dev.tsv'
prompt_tst_path = '../../atomic2020_data-feb2021/prompt_datasets/tst.tsv'

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def read_tsv_lines(input_file: str) -> List[dict]:
    # """
    # parse line to dict, each line from tsv file follows "<head> @@ <relation> <tab> <tail_1>|<tail_2>|..." format
    # """
    re = []
    # with open(input_file) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         line = line.split('\t')
    #         re.append( {"head_event": line[0].split('@@')[0].strip(), 
    #                     "relation": line[0].split('@@')[1].strip(),
    #                     "tail_event": [l.strip() for l in line[1].strip('\n').split('|')]})
    data = pd.read_csv(input_file, sep='\t')
    for idx, row in data.iterrows():
        re.append({"head_event": row.head_event, "relation": row.relation, "tail_event": [t.strip() for t in row.tail_event.strip('\n').split('|')]})
    return re

def write_items(output_file, items):
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()

def preprocessing(dataset: DataFrame, config, filter_none=False):
    if filter_none:
        # logger.info("before: {}".format(len(dataset)))
        logger.info("remove samples of which the tail event is none")
        dataset = dataset[dataset['tail_event']!='none']
        # logger.info("after: {}".format(len(dataset)))
    if not config.USE_PROMPT:
        logger.info("use <special token> to represent relations")
        dataset.head_event = dataset.head_event + ' ' + dataset.relation + " [GEN]"
        dataset.tail_event = dataset.tail_event + ' [EOS]'
    else:
        logger.info("use manual prompt to represent relations")
        if 'prompt' not in dataset.columns:
            prompts = []
            for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
                prompts.append(manual_prompt("atomic2020", row['head_event'], row['relation'], row['tail_event']))
            dataset['prompt'] = prompts

        dataset.head_event = dataset.prompt + " [GEN]"
        dataset.tail_event = dataset.tail_event + ' [EOS]'
    for i in range(3):
        logger.info(dataset.head_event.tolist()[i])
    return dataset.reset_index(drop=True)

def postprocessing(tokenizer, pred_generations: list, test_data: pd.DataFrame):
    """
    first truncate generations, then combine each generation and its corresponding references for the use in automatic evaluation 
    """
    assert len(pred_generations)==len(test_data), "the length of pred_generations: {} and the length of test_data: {} do not match".format(len(pred_generations), len(test_data))
    error_cnt = 0
    heads = test_data.head_event.tolist()
    rels = test_data.relation.tolist()
    refs = test_data.tail_event.tolist()
    re = []
    for idx, item in enumerate(pred_generations):
        gens = []
        for gen in item['generations']:
            gen_idx = gen.index('[GEN]')
            # try:
            #     eos_idx = gen.index(tokenizer.eos_token)
            # except:
            #     error_cnt += 1
            #     continue
            gens.append(gen[gen_idx+5:].split(tokenizer.eos_token)[0].strip())
        re.append({ "generation": gens, 
                    "references": refs[idx], 
                    "input": {"head": heads[idx], "relation": rels[idx]}
                    })
    # logger.info(f"found {error_cnt} generations without [EOS] token")
    return re

def fewshot_tuning(config, model):
    """
    fine-tuning model on few-shot relations
    """



def main():
    wandb.init(project="gpt2_comet_atomic")
    config = wandb.config
    config.TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 16))
    config.VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE", 2))
    config.ACCUM_STEP = int(os.environ.get("ACCUM_STEP", 2))
    config.TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 1))
    config.VAL_EPOCHS = int(os.environ.get("VAL_EPOCHS", 1))
    config.FEWSHOT_EPOCHS = int(os.environ.get("FEWSHOT_EPOCHS", 3))
    config.LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
    config.SEED = int(os.environ.get("SEED", 24))
    config.IN_LEN = int(os.environ.get("IN_LEN", 48))
    config.OUT_LEN = int(os.environ.get("OUT_LEN", 64))
    config.SUMMARY_LEN = 0 # Used for t5
    config.OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output/")
    config.OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "output.jsonl")
    config.MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_PATH", "./models/")
    config.DO_TRAIN = os.environ.get("DO_TRAIN", "False") == "True"
    config.DO_PRED = os.environ.get("DO_PRED", "True") == "True"
    config.PRED_FILE = str(os.environ.get("PRED_FILE", "../../atomic2020_data-feb2021/fewshot_test1.tsv"))
    config.TOP_K = int(os.environ.get("TOP_K", 40)) 
    config.PRED_BATCH = 64
    config.TOKENIZER = os.environ.get('TOKENIZER', "gpt2-xl")

    config.USE_PROMPT = os.environ.get('USE_PROMPT', "False") == "True"
    config.DO_FEWSHOT = os.environ.get('DO_FEWSHOT', "False") == "True"
    config.EXTRA_RELS = os.environ.get('EXTRA_RELS', "False") == "True"
    config.N_SHOT = int(os.environ.get("N_SHOT", 10))

    if config.DO_FEWSHOT:
        d = dict(config)
        d['USE_PROMPT'] = True
        d['DO_PRED'] = False
        config.update(d, allow_val_change=True)

    logger.info(config)
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    model_name = "gpt2-xl" if 'GPT2_MODEL' not in os.environ else os.environ['GPT2_MODEL']

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(config.TOKENIZER)
    if not config.USE_PROMPT:
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                'LocationOfAction',
                'HinderedBy',
                'HasFirstSubevent',
                'NotHasProperty',
                'NotHasA',
                'HasA',
                'AtLocation',
                'NotCapableOf',
                'CausesDesire',
                'HasPainCharacter',
                'NotDesires',
                'MadeUpOf',
                'InstanceOf',
                'SymbolOf',
                'xReason',
                'isAfter',
                'HasPrerequisite',
                'UsedFor',
                'MadeOf',
                'MotivatedByGoal',
                'Causes',
                'oEffect',
                'CreatedBy',
                'ReceivesAction',
                'NotMadeOf',
                'xWant',
                'PartOf',
                'DesireOf',
                'HasPainIntensity',
                'xAttr',
                'DefinedAs',
                'oReact',
                'xIntent',
                'HasSubevent',
                'oWant',
                'HasProperty',
                'IsA',
                'HasSubEvent',
                'LocatedNear',
                'Desires',
                'isFilledBy',
                'isBefore',
                'InheritsFrom',
                'xNeed',
                'xEffect',
                'xReact',
                'HasLastSubevent',
                'RelatedTo',
                'CapableOf',
                'NotIsA',
                'ObjectUse',
            ]
        })
    tokenizer.add_special_tokens({'eos_token': '[EOS]','pad_token': '[PAD]', 'additional_special_tokens': ['GEN']})
    
    logging.info("Loading model from {}".format(model_name))
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    if use_parallel and not (not config.DO_TRAIN and config.DO_PRED):
        model = nn.DataParallel(model)
        model = model.to(device)
        logging.info("Move model to {} devices".format(torch.cuda.device_count()))
        optimizer = torch.optim.Adam(params=model.module.parameters(), lr=config.LEARNING_RATE)
    else:
        logging.info("Move model to device {}".format(device))
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    wandb.watch(model, log="all")

    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    fewshot_params = {
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
    }


    if config.DO_TRAIN:
        logger.info('Initiating Fine-Tuning for the model on our dataset')

        # load datasets
        global trn_data_path, dev_data_path, tst_data_path
        if config.USE_PROMPT:
            trn_data_path, dev_data_path, tst_data_path = prompt_trn_path, prompt_dev_path, prompt_tst_path
            
        train_dataset = pd.read_csv(os.environ.get('TRAIN_DATA_PATH', trn_data_path), encoding='latin-1', sep="\t")
        val_dataset = pd.read_csv(os.environ.get('DEV_DATA_PATH', dev_data_path), encoding='latin-1', sep="\t")
        test_dataset = pd.read_csv(os.environ.get('TEST_DATA_PATH', tst_data_path), encoding='latin-1', sep="\t")
        val_dataset_mini = pd.read_csv(os.environ.get('DEV_DATA_PATH', dev_data_path), encoding='latin-1', sep="\t")
        val_dataset_mini = val_dataset_mini.sample(n=min(int(val_dataset_mini.size / 3), 100),
                                                random_state=config.SEED)
        # remove few-shot relations
        if FEW_SHOT_RELATIONS:
            train_dataset = train_dataset[~train_dataset['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
            val_dataset = val_dataset[~val_dataset['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
            test_dataset = test_dataset[~test_dataset['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
            val_dataset_mini = val_dataset_mini[~val_dataset_mini['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
            logger.info("removed tuples of relation: {}".format(FEW_SHOT_RELATIONS))

        train_dataset = preprocessing(train_dataset, config, FILTER_NONE)
        val_dataset = preprocessing(val_dataset, config, FILTER_NONE)
        test_dataset = preprocessing(test_dataset, config, FILTER_NONE)
        val_dataset_mini = preprocessing(val_dataset_mini, config, FILTER_NONE)
        

        logger.info("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
        logger.info("DEV Dataset tuple_count: {}".format(val_dataset.shape))
        logger.info("DEV MINI Dataset tuple_count: {}".format(val_dataset_mini.shape))

        training_set = KGDataset(train_dataset, tokenizer, config.OUT_LEN, config.SUMMARY_LEN, model="gpt2")
        val_set = KGDataset(val_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        val_set_mini = KGDataset(val_dataset.head(2000), tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        test_set = KGDataset(test_dataset, tokenizer, config.IN_LEN,  config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)


        training_loader = DataLoader(training_set, **train_params, drop_last=True)
        val_loader = DataLoader(val_set, **val_params, drop_last=True)
        test_loader = DataLoader(test_set, **val_params, drop_last=True)
        val_loader_mini = DataLoader(val_set_mini, **val_params, drop_last=True)

        for epoch in range(config.TRAIN_EPOCHS):
            save_dir = 'models' if not config.USE_PROMPT else 'models/prompt_models/'
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader_mini, model_class="gpt2", save_dir=save_dir, accum_step=config.ACCUM_STEP)
            save_dir = '{}/checkpoint_{}'.format(config.MODEL_SAVE_DIR, epoch) if not config.USE_PROMPT else '{}/prompt_models/checkpoint_{}'.format(config.MODEL_SAVE_DIR, epoch)
            save_model(model, save_dir)
            tokenizer.save_pretrained(save_dir)
        # save_model(model, '/models')

    if config.DO_FEWSHOT:
        # finetune model on few-shot relations
        assert FEW_SHOT_RELATIONS != [], "FEW_SHOT_RELATIONS should not be empty if want to do few-shot tuning"
        trn_data_path = prompt_trn_path  
        train_data = pd.read_csv(os.environ.get('TRAIN_DATA_PATH', trn_data_path), encoding='latin-1', sep="\t")
        train_dataset = pd.DataFrame()
        for rel in FEW_SHOT_RELATIONS:
            train_dataset = pd.concat([train_dataset, train_data[train_data['relation']==rel].sample(n=config.N_SHOT, random_state=config.SEED)])
        train_dataset = train_dataset.reset_index(drop=True)
        train_dataset = preprocessing(train_dataset, config, FILTER_NONE)
        logger.info("TRAIN Dataset tuple count: {}".format(train_dataset.shape))
        training_set = KGDataset(train_dataset, tokenizer, config.OUT_LEN, config.SUMMARY_LEN, model="gpt2")
        training_loader = DataLoader(training_set, **fewshot_params, drop_last=True)

        for epoch in range(config.FEWSHOT_EPOCHS):
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader=None, model_class="gpt2", save=False, accum_step=config.ACCUM_STEP)
        # run generation
        records = read_tsv_lines(config.PRED_FILE)
        # records = records[:10]
        pred_data = pd.DataFrame.from_records(records)
        pred_data = pred_data[pred_data['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
        pred_dataset = pred_data.explode('tail_event')
        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)
        pred_dataset = preprocessing(pred_dataset, config)
        logger.info("PRED data tuple count: {}".format(len(pred_dataset)))

        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K)
        pred_generations = postprocessing(tokenizer, pred_generations, pred_data)

        write_items(os.path.join(config.OUTPUT_DIR, config.OUTPUT_NAME),
                    [json.dumps(r) for r in pred_generations])


    if config.DO_PRED:

        if config.PRED_FILE.endswith("jsonl"):
            records = read_jsonl_lines(config.PRED_FILE)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        elif config.PRED_FILE.endswith("tsv"):
            records = read_tsv_lines(config.PRED_FILE)
            # records = records[:10]
            pred_data = pd.DataFrame.from_records(records)
            if FEW_SHOT_RELATIONS:
                pred_data = pred_data[pred_data['relation'].isin(FEW_SHOT_RELATIONS)].reset_index(drop=True)
            pred_dataset = pred_data.explode('tail_event')
        else:
            pred_dataset = pd.read_csv(config.PRED_FILE, encoding='latin-1', sep="\t")



        pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)
        pred_dataset = preprocessing(pred_dataset, config)



        pred_set = KGDataset(pred_dataset, tokenizer, config.IN_LEN, config.OUT_LEN - config.IN_LEN, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, top_k=config.TOP_K)

        pred_generations = postprocessing(tokenizer, pred_generations, pred_data)

        write_items(os.path.join(config.OUTPUT_DIR, config.OUTPUT_NAME),
                    [json.dumps(r) for r in pred_generations])

        # # Resave the model to keep generations and model associated
        # model.save_pretrained('/models')
        # tokenizer.save_pretrained('/models')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-t", "--test_install",
                      action="store_true", default=False,
                      help="Test install, without running any modeling code.")

    (options, args) = parser.parse_args()
    if not options.test_install:
        main()
