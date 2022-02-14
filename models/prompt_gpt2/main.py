# Importing stock libraries
import enum
import json
from typing import List
import argparse
import pickle

import numpy as np
import pandas as pd

import torch
from torch.cuda import device
import torch.nn as nn
from torch.nn import parallel


# Importing the T5 modules from huggingface/transformers
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup


from OpenPrompt.openprompt.plms import load_plm
from OpenPrompt.openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate, PrefixTuningTemplate
from OpenPrompt.openprompt import PromptDataLoader
from OpenPrompt.openprompt import PromptForGeneration

import os
from tqdm import tqdm
import time
import logging
logger = logging.getLogger()

import sys
sys.path.append('..')
sys.path.append('../..')



from prompts import manual_template_str, promptTuning_template_str 
from dataset import PromptDatasetProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
device_cnt = torch.cuda.device_count()
device_info = 'use cpu' if device =='cpu' else f'use cuda | num: {device_cnt}'
print(device_info)

METHODS = ['manual', 'soft', 'prefixTuning']

RELATIONS = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires',
             'isAfter', 'HasSubEvent', 'isBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy', 
             'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant']

# RELATIONS = ['MadeUpOf']

prompt_trn_path = '../../atomic2020_data-feb2021/prompt_datasets/train.tsv'
prompt_dev_path = '../../atomic2020_data-feb2021/prompt_datasets/dev.tsv'
prompt_tst_path = '../../atomic2020_data-feb2021/prompt_datasets/tst.tsv'
pred_data_path = '../../atomic2020_data-feb2021/sampled_test.tsv'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='gpt2-xl')
parser.add_argument('--model_save_dir', default='./models')
parser.add_argument('--prompt_load_path', default="")
parser.add_argument('--prompt_save_dir', default='./prompt_encoders')
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--method', default="manual", 
                    help=f"one of {METHODS}")
parser.add_argument('--soft_len', type=int, default=100,
                    help = "number of soft token used in building continuous prompt templates")

parser.add_argument('--data_dir', default='../../atomic2020_data-feb2021/sep_relations')
parser.add_argument('--output_dir', default='./output')
parser.add_argument('--output_file_name', default="",
                    help="you can specify the generation file name instead of default name")
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_pred', action= 'store_true')
parser.add_argument('--n_shot', type=int, default=None, 
                    help="number of train/val samples per relation")
parser.add_argument('--rel', default="",
                    help="you can specify certain relation in {}".format(RELATIONS))

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--pred_batch_size', type=int, default=1)
parser.add_argument('--input_len', type=int, default=192)
parser.add_argument('--output_len', type=int, default=192)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--top_p', type=int, default=0.9)
parser.add_argument('--num_beams', type=int, default=10)
parser.add_argument('--num_sequences', type=int, default=10)
parser.add_argument('--stop_token', default='.')


parser.add_argument('--epoch', type=int, default=None)
parser.add_argument("--lr", type=float, default=0.3)
parser.add_argument("--early_stop", type=int, default=8)
parser.add_argument("--total_steps", type=int, default=30000)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--ckpt_itvl", type=int, default=2000,
                    help = "step interval for checkpoint")
# parser.add_argument("--decay_rate", type=float, default=1) # constant lr
# parser.add_argument("--steps_per_decay", type=int, default=8000)
parser.add_argument('--log', action='store_true', help="set to true if want to save specific log file")




def write_items(items: List[str], output_file):
    fdir = os.path.dirname(output_file)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()

def init_template(method, plm, tokenizer, soft_len=0, rel=None):
    soft = "{'soft'}"
    prompt = "{'meta': 'prompt'}"
    mask = "{'mask'}"
    if method == "manual":
        text = manual_template_str()[rel]
        logger.info("init manual template: {}".format(text))
        return ManualTemplate(tokenizer=tokenizer, text=text)
    elif method == 'soft':
        # text = promptTuning_template_str(soft_len)[rel]
        text = f"{soft*soft_len} {prompt} {mask}"
        logger.info("init prompt-tuning template: {}".format(text))
        return MixedTemplate(model=plm, tokenizer=tokenizer, text=text)
    elif method == "prefixTuning":
        logger.info("init prfix-tuning template with {} soft tokens".format(soft_len))
        return PrefixTuningTemplate(model=plm, tokenizer=tokenizer, num_token=soft_len)


def save_prompt_encoder(template, path):
    dir_ = os.path.dirname(path)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    with open(path, 'wb') as f:
        pickle.dump(template, f)
    logger.info(f'save prompt encoder to {path}')

def load_prompt_enoder(path):
    with open(path, 'rb') as f:
        template = pickle.load(f)
    logger.info(f"load prompt encoder from {path}")
    return template

def evaluate(prompt_model, data_loader):
    tot_loss = 0
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs.to(device)
            loss = prompt_model(inputs)
            tot_loss += loss.item()
    return tot_loss / len(data_loader)

def prompt_train(args, prompt_model, trn_loader, val_loader, tst_loader, tokenizer=None):
    global_step = 0 
    tot_loss = 0 
    log_loss = 0
    best_val_loss = None
    cnt = 0
    itvl = args.ckpt_itvl
    save_path = os.path.join(args.prompt_save_dir, args.model_name + '_' + args.method + '_PromptEncoder.pkl')

    if device_cnt > 1:
        parallel_obj, prompt_model = prompt_model, prompt_model.module

    # params = [{'params': prompt_model.template.parameters()}]
    grouped_parameters = [
        {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=args.lr)
    if args.epoch:
        tot_epoch = args.epoch
        tot_step = len(trn_loader)*args.epoch
    else:
        tot_step = args.total_steps
        tot_epoch = int(tot_step / (len(trn_loader)/args.accum_step))
    logger.info("total steps: {} total epoches: {}".format(tot_step, tot_epoch))

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, tot_step)

    for epoch in range(tot_epoch):
        logger.info(f"==================== {args.rel} Epoch {epoch} ====================")
        # train    
        prompt_model.train()
        for step, inputs in tqdm(enumerate(trn_loader), total=len(trn_loader), desc='training'):
            inputs.to(device)
            loss = prompt_model(inputs)

            if step % 1000 == 0:
                logger.info("tot_step: {} trn loss: {:.3f} lr: {:.5f}".format(global_step, loss.item(),scheduler.get_last_lr()[0]))

            # loss regularization
            loss /= args.accum_step
            loss.backward()
            tot_loss += loss.item()

            if (step+1) % args.accum_step == 0:
                # torch.nn.utils.clip_grad_norm_(prompt_model.template.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
            # if not args.freeze_scheduler:
            #     scheduler.step()
                
            if global_step > 0 and global_step % itvl == 0:
                val_loss = evaluate(prompt_model, val_loader)
                # tst_loss= evaluate(prompt_model, tst_loader)
                # logger.info("tot_step: {} trn: {:.3f} dev: {:.3f} tst: {:.3f} lr: {:.5f}".format(global_step, (tot_loss-log_loss)/itvl, val_loss, tst_loss, scheduler.get_last_lr()[0]))
                logger.info("tot_step: {} trn: {:.3f} dev: {:.3f} lr: {:.5f}".format(global_step, (tot_loss-log_loss)/itvl, val_loss, scheduler.get_last_lr()[0]))
                log_loss = tot_loss
                if not best_val_loss or best_val_loss > val_loss:
                    best_val_loss = val_loss
                    save_prompt_encoder(prompt_model.template, save_path)
                    cnt = 0
                else: cnt += 1
            if global_step >= tot_step or cnt >= args.early_stop:
                logger.info("==================== finish training ====================")
                return
        

    
def proc_dataloader(data_loader, input_len):
    cnt = 0
    for idx, tensor_data in tqdm(enumerate(data_loader.tensor_dataset), total=len(data_loader.tensor_dataset)):
        if len(tensor_data['input_ids']) > input_len:
            data_loader.wrapped_dataset.pop(idx)
            data_loader.tensor_dataset.pop(idx)
            cnt += 1
    logger.info("dropped {} data".format(cnt))


def postprocess(text, stop_token):
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start if start != -1 else None
    text = text[:find_nth(text, stop_token, 1)]
    text = text.strip('\n')
    text = text.strip(stop_token)
    # text = text[:find_nth(text, stop_token, 1)] if stop_token not in prompt else text[:find_nth(text, stop_token, 2)]
    return text

def run_generation(args, prompt_model, dataloader, pred_dataset, tokenizer):
    re = []
    generation_arguments = {
        # "max_length": args.output_len,
        "max_new_tokens": None,
        "temperature": 1.0,
        "do_sample": True,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": 1.0,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_sequences,
        # "bad_words_ids": [[628], [198]]
    }
    data_idx = 0
    if device_cnt > 1:
        parallel_obj = prompt_model
        prompt_model = parallel_obj.module
    for idx, inputs in tqdm(enumerate(dataloader), total=len(pred_dataset)//args.pred_batch_size):
        inputs.to(device)
        input_len = max(inputs['input_ids_len'])
        generation_arguments['max_length'] = input_len + 10
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        for i in range(len(inputs['input_ids'])):
            prompt = tokenizer.decode(inputs["input_ids"][i][:inputs['input_ids_len'][i]])
            generation = output_sentence[i]
            generation = postprocess(generation, ".")
            item = {
                        'generation': [generation],
                        'references': [t.strip() for t in inputs['tgt_text'][i].strip('\n').split('|')],
                        'input': {'head': pred_dataset[data_idx].text_a, 'relation': pred_dataset[data_idx].text_b, 'prompt': prompt}
                    }
            re.append(item)
            data_idx += 1
            if idx % 10 == 0:
                print(item)
    assert data_idx == len(pred_dataset), f"data_idx: {data_idx}, len(pred_dataset): {len(pred_dataset)}"
    return re

def main():
    args = parser.parse_args()
    #assert os.path.exists(args.prompt_name_or_dir) or args.prompt_name_or_dir in TEMPLATE_CLASSES.keys(), f"args.prompt_name_or_dir should be either existing path or one of {TEMPLATE_CLASSES.keys()}, but got {args.prompt_name_or_dir}"
    assert args.method in METHODS, f"args.method should be in {METHODS}, but got {args.method}"
    assert args.do_train or args.do_pred, "both do_train and do_pred set to false, nothing to do"

    # args.freeze_scheduler = (args.method == 'soft')

    # configuring logger
    if args.log:
        log_fname = os.path.join(args.log_dir, os.path.basename(args.model_name_or_path)+'_'+args.method+'_'+time.strftime("%m-%d_%H-%M", time.localtime())+'.log')
    else:
        log_fname = 'tmp.log'
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_fname,'w'),
        logging.StreamHandler()
        ]
    )
    logger.info(args)
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True



    global RELATIONS
    if args.rel in RELATIONS:
        RELATIONS = [args.rel]

    if 'gpt2' in args.model_name_or_path:
        args.model_type = 'gpt2'
    elif 't5' in args.model_name_or_path:
        args.model_type = 't5-lm'
    model_name = os.path.basename(args.model_name_or_path)
    args.model_name = model_name
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_type, args.model_name_or_path)

    # bug fix
    if args.model_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
        # # WrapperClass.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        plm.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.method == 'manual':
            logger.info("manual template, no parameters to train")
    elif args.do_train:
        # for rel in RELATIONS:
            # args.rel = rel
        # build prompt template    
        template = init_template(method=args.method, plm=plm, tokenizer=tokenizer, soft_len=args.soft_len)  
        # logger.info(template)
        # load data

        try:
            with open('./input/trn_dataloader.pkl', 'rb') as f:
                trn_dataloader = pickle.load(f)
            logger.info("loaded train data from ./input/trn_dataloader.pkl")
        except:
            logger.info(f"loading train data...")
            trn_dataset = PromptDatasetProcessor().get_examples(prompt_trn_path)
            logger.info("train example: {}".format(trn_dataset[0]))

            if args.n_shot:
                trn_dataset = np.random.choice(trn_dataset, size=args.n_shot).tolist()
                logger.info("sampled {} examples in dataset".format(args.n_shot))
            
            
            trn_dataloader = PromptDataLoader(dataset=trn_dataset, template=template, tokenizer=tokenizer, 
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=args.input_len, decoder_max_length=args.output_len, 
                                            batch_size=args.batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
                                            truncate_method="head")
            proc_dataloader(trn_dataloader, args.input_len)
            logger.info("wrapped train example: {}".format(trn_dataloader.wrapped_dataset[0]))

            with open('./input/trn_dataloader.pkl', 'wb') as f:
                pickle.dump(trn_dataloader, f)
            logger.info("dump trn_dataloader to ./input/trn_dataloader.pkl")

        logger.info(f"loading val data...")
        val_dataset = PromptDatasetProcessor().get_examples(prompt_dev_path)
        if args.n_shot:
            val_dataset = np.random.choice(val_dataset, size=args.n_shot).tolist()
            logger.info("sampled {} examples in dataset".format(args.n_shot))            

        val_dataloader = PromptDataLoader(dataset=val_dataset, template=template, tokenizer=tokenizer, 
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.input_len, decoder_max_length=args.output_len, 
                                        batch_size=args.batch_size, shuffle=False, teacher_forcing=True, predict_eos_token=True,
                                        truncate_method="head")
        proc_dataloader(val_dataloader, args.input_len)

        logger.info(f"loading test data...")
        tst_dataset = PromptDatasetProcessor().get_examples(prompt_tst_path)

        if len(tst_dataset) > len(val_dataset):
            tst_dataset = np.random.choice(tst_dataset, size=len(val_dataset)).tolist()
            logger.info("sampled {} examples in dataset".format(len(tst_dataset)))         
        
        tst_dataloader = PromptDataLoader(dataset=tst_dataset, template=template, tokenizer=tokenizer, 
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.input_len, decoder_max_length=args.output_len, 
                                        batch_size=args.batch_size, shuffle=False, teacher_forcing=True, predict_eos_token=True,
                                        truncate_method="head")
        proc_dataloader(tst_dataloader, args.input_len)

        # load the pipeline model PromptForGeneration.
        prompt_model = PromptForGeneration(plm=plm,template=template, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=False)
        if device_cnt > 1:
            prompt_model = nn.DataParallel(prompt_model)
        prompt_model.to(device)
        prompt_train(args, prompt_model, trn_dataloader, val_dataloader, tst_dataloader, tokenizer)



    if args.do_pred:
        generations_for_fact = []
        # run generation on all relations
        # for rel in RELATIONS:
            # args.rel = rel
        logger.info(f"loading pred data...")
        do_analysis = (args.method == 'manual' or args.method == 'soft')
        pred_dataset = PromptDatasetProcessor().get_examples(pred_data_path, analysis=do_analysis, skip_none=False)

        # pred_dataset = pred_dataset[:5]

    
        # build prompt template
        if args.prompt_load_path:
            template = load_prompt_enoder(args.prompt_load_path)
        else:  
            template = init_template(method=args.method, plm=plm, tokenizer=tokenizer, soft_len=args.soft_len)  

        pred_dataloader = PromptDataLoader(dataset=pred_dataset, template=template, tokenizer=tokenizer, 
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=args.input_len, decoder_max_length=args.output_len, 
                                            batch_size=args.pred_batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
                                            truncate_method="head", padding=False) 
        # load the pipeline model PromptForGeneration.
        prompt_model = PromptForGeneration(plm=plm,template=template, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=True)
        if device_cnt > 1:
            prompt_model = nn.DataParallel(prompt_model)
        prompt_model.to(device)
        # evaluating
        logger.info(f"===== Evaluating {model_name} model =====")
        prompt_model.eval()
        generations_for_fact.extend(run_generation(args, prompt_model, pred_dataloader, pred_dataset, tokenizer))


        # save generation
        fname = args.output_file_name if args.output_file_name != "" else f"{args.method}-prompt-{model_name}-gen.jsonl"
        write_items([json.dumps(r) for r in generations_for_fact], os.path.join(args.output_dir, fname))
        logger.info(f"save {fname} to {args.output_dir}")
        
    # save model
    # model_save_path = os.path.join(args.model_save_dir, fname.strip('.jsonl'))
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    # model.save_pretrained(model_save_path)
    # tokenizer.save_pretrained(model_save_path)
    
if __name__ == "__main__":
    main()