import argparse

project_name = 'InferTextGen'
tags = ['prompt']

METHODS = ['manual', 'soft', 'prefixTuning']

RELATIONS = ['ObjectUse', 'AtLocation', 'MadeUpOf', 'HasProperty', 'CapableOf', 'Desires', 'NotDesires',
             'isAfter', 'HasSubEvent', 'isBefore', 'HinderedBy', 'Causes', 'xReason', 'isFilledBy', 
             'xNeed', 'xAttr', 'xEffect', 'xReact', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant']

prompt_trn_path = '../../atomic2020_data-feb2021/prompt_datasets/train.tsv'
prompt_dev_path = '../../atomic2020_data-feb2021/prompt_datasets/dev.tsv'
prompt_tst_path = '../../atomic2020_data-feb2021/prompt_datasets/tst.tsv'
pred_data_path = '../../atomic2020_data-feb2021/sampled_test.tsv'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='', help='can specify experiment name, used in wandb experiment records',) 
    # dirs and paths
    parser.add_argument('--model_name_or_path', default='gpt2-xl')
    parser.add_argument('--model_save_dir', default='./models')
    parser.add_argument('--prompt_load_path', default="")
    parser.add_argument('--prompt_save_dir', default='./prompt_encoders')
    parser.add_argument('--prompt_save_name', default="", 
                        help="you can specify the prompt embedding file name instead of default name")
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--data_dir', default='../../atomic2020_data-feb2021/sep_relations')
    parser.add_argument('--output_dir', default='../../output')
    parser.add_argument('--output_file_name', default="",
                        help="you can specify the generation file name instead of default name")

    parser.add_argument('--method', default="manual", 
                        help=f"one of {METHODS}")
    parser.add_argument('--soft_len', type=int, default=100,
                        help = "number of soft token used in building continuous prompt templates")

    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_pred', action= 'store_true')
    parser.add_argument('--n_shot', type=int, default=None, 
                        help="number of train/val samples per relation")
    parser.add_argument('--rel', default="",
                        help="you can specify certain relation in {}".format(RELATIONS))
    parser.add_argument('--skip_none', action='store_true', help='remove all none-tail tuples in train/dev/test datasets')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accum_step', type=int, default=4)
    parser.add_argument('--pred_batch_size', type=int, default=1)
    parser.add_argument('--input_len', type=int, default=192)
    parser.add_argument('--output_len', type=int, default=192)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=int, default=0.9)
    parser.add_argument('--num_beams', type=int, default=10)
    parser.add_argument('--num_sequences', type=int, default=5)
    parser.add_argument('--stop_token', default='.')


    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--scheduler", choices=['linear', 'constant'], default='linear')
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--ckpt_itvl", type=int, default=1000,
                        help = "step interval for checkpoint")
    # parser.add_argument("--decay_rate", type=float, default=1) # constant lr
    # parser.add_argument("--steps_per_decay", type=int, default=8000)
    parser.add_argument('--log', action='store_true', help="set to true if want to save specific log file")

    args = parser.parse_args()
    return args