import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append('../')
sys.path.append('../../')
from gpt2_zeroshot import manual_prompt


trn_data_path = '../../atomic2020_data-feb2021/train.tsv'
dev_data_path = '../../atomic2020_data-feb2021/dev.tsv'
tst_data_path = '../../atomic2020_data-feb2021/test.tsv'

prompt_trn_path = '../../atomic2020_data-feb2021/prompt_datasets/train.tsv'
prompt_dev_path = '../../atomic2020_data-feb2021/prompt_datasets/dev.tsv'
prompt_tst_path = '../../atomic2020_data-feb2021/prompt_datasets/tst.tsv'


paths = [prompt_trn_path, prompt_dev_path, prompt_tst_path]

if __name__ == '__main__':
        # load datasets
    train_dataset = pd.read_csv(os.environ.get('TRAIN_DATA_PATH', trn_data_path),encoding='latin-1', sep="\t")
    val_dataset = pd.read_csv(os.environ.get('DEV_DATA_PATH', dev_data_path), encoding='latin-1', sep="\t")
    test_dataset = pd.read_csv(os.environ.get('TEST_DATA_PATH', tst_data_path), encoding='latin-1', sep="\t")

    for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
        prompts = []
        for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
            re = manual_prompt("atomic2020", row['head_event'], row['relation'], row['tail_event'])
            if idx < 10:
                print(re)
            prompts.append(re)
        dataset['prompt'] = prompts
        print("save to {}".format(paths[i]))
        dataset.to_csv(paths[i], encoding='latin-1', sep='\t', index=False)
        