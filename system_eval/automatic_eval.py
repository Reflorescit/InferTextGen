import argparse
import pandas as pd
from utils import read_jsonl, write_jsonl
from evaluation.eval import QGEvalCap
from tabulate import tabulate
import os

import logging
logger = logging.getLogger("autoeval")
logging.basicConfig(level=logging.INFO)

import pdb

RELATIONS = []
# RELATIONS = ['MadeUpOf', 'NotDesires', 'xReason', 'isBefore', 'xReact', 'oWant']
# RELATIONS = ['ObjectUse', 'CapableOf', 'AtLocation', 'xAttr']

def get_tuple(l):
    gens = l['generation'] if type(l["generation"])==list else  [l["generation"]]
    head = l["input"]["head"]
    tails = [ref.strip() for ref in l["references"]]
    relation = l["input"]["relation"]
    return {"head": head, "relation": relation, "tails": tails, "generations": gens}


def topk_eval(model_name, data, k):

    topk_gts = {}
    topk_res = {}
    instances = []
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []

    for i, l in enumerate(data):
        t = get_tuple(l)
        gens = t["generations"]
        tails = t["tails"]
        head = t["head"]
        for (j, g) in enumerate(gens[:k]):

            instance = t.copy()
            instance["generation"] = g
            instances.append(instance)

            key = str(i) + "_" + str(j)
            topk_gts[key] = tails
            topk_res[key] = [g]

            if g in tails:
                topk_exact_match.append((l, 1))
                if g != "none":
                    topk_exact_match_not_none.append((l, 1))
            else:
                topk_exact_match.append((l, 0))
                if g != "none":
                    topk_exact_match_not_none.append((l, 0))
            if g == head:
                topk_is_head.append((l, 1))
            else:
                topk_is_head.append((l, 0))

    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    score, scores = QGEval.evaluate()
    
    return score, scores, instances


def eval(data_file, model_name, args):
    data = read_jsonl(data_file)
    if len(data) == 0:
        return None
    if args.relations:
        data = [item for item in data if item['input']['relation'] in args.relations]
    if args.filter_none:
        for idx, item in enumerate(data):
            data[idx]['references'] = [item for item in data[idx]['references'] if item != 'none']
        data = [item for item in data if item['references']]
    return topk_eval(model_name, data, k=args.k)

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

def Row2CSV(rows):
    data_dict = {}
    index = [r[0] for r in rows[1:]]
    for i, name in enumerate(rows[0][1:]):
        data_dict[name] = [r[1:][i] for r in rows[1:]]
    df = pd.DataFrame(data_dict, index=index)
    return df
    



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_file', type=str, help='Results file on ATOMIC2020 test set')
    parser.add_argument('--input_file', nargs="+", help="generation file path, multiple paths seperated with space")
    parser.add_argument('--exclude_none', action='store_true', help="do not evaluate empty generation")
    parser.add_argument('--output_dir', type=str, default="../../output/scores")
    parser.add_argument('--output_name', type=str, default="score_table.csv")
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--filter_none', action='store_true', help='remove none in all references')
    parser.add_argument('--sep_rel', action='store_true', help="compute separate scores for each relation")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # generations_file = preprocess_generations(args)
    # input_file = generations_file

    input_files = [os.path.abspath(path) for path in args.input_file] 


    expts = [
        [input_file,  os.path.basename(input_file).split('.')[0]] for input_file in input_files
    ]

    global RELATIONS
    relation_groups = [[rel] for rel in RELATIONS] if RELATIONS and args.sep_rel else [RELATIONS]

    for rel_group in relation_groups:
        args.relations = rel_group
        scores_per_model = []
        add_column = True
        for f, m in expts:
            # result_file = './results/{}_scores.jsonl'.format(m)
            # result_file = os.path.join(args.output_dir, '{}_scores.jsonl'.format(m))

            s, scores, instances = eval(f, model_name=m, args=args)
            if s == None:
                logger.info("Skipping " + m)
                continue


            for k in scores.keys():
                assert len(scores[k]) == len(instances)

            #results = {"model": m, "scores": s, "all_scores": scores, "instances": instances}
            results = {"model": m, "scores": s}
            # write_jsonl(result_file, [results])
            # logger.info("scores written to {}".format(result_file))

            scores_per_model.append(results)
            columns = list(results["scores"].keys())
            s_row = toRow(results["model"], results["scores"], columns)
            if add_column:
                rows = [[""] + columns]
                add_column = False
            rows.append(s_row)

        # import datetime
        # date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        #logger.info(scores_per_model)

        # write_jsonl('./results/scores_{}.jsonl'.format(date), scores_per_model)
        #logger.info(tabulate(rows, headers='firstrow', tablefmt='latex', floatfmt='#.3f'))
        if rel_group:
            logger.info(f"\n---------- {rel_group} ----------")
        table = tabulate(rows, tablefmt='tsv', floatfmt='#.3f')
        logger.info('\n'+table)
        output_path=os.path.join(args.output_dir, args.output_name)
        df = Row2CSV(rows)
        df.to_csv(output_path)
        logger.info("save score files to {}".format(output_path))

if __name__ == "__main__":
    main()
