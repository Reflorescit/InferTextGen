import json
import sys
import csv
import operator
import random
import os

from nltk.translate.bleu_score import sentence_bleu

def get_reference_sentences(filename):
    result = []
    with open(filename) as file:
        for line in file:
            result.append([x.strip() for x in line.split('\t')[1].split('|')])
    return result

def postprocess(sentence):
    return sentence

def get_heads_and_relations(filename):
    result = []
    with open(filename) as file:
        for line in file:
            line = line.split('\t')[0]
            head_event = line.split('@@')[0].strip()
            relation = line.split('@@')[1].strip()
            to_add = {
                'head': head_event,
                'relation': relation
            }
            result.append(to_add)
    return result

def get_hypothesises(filename):
    result = []
    import json

    with open(filename) as file:
        for line in file:
            result.append(json.loads(line)["greedy"])
    return result

def preprocess_generations(args):
    input_file = args.input_file

    outfile_path = os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0] + "_gens.jsonl")

    outfile = open(outfile_path, 'w')

    references_list = get_reference_sentences('test.tsv')
    heads_relations = get_heads_and_relations('test.tsv')
    hypothesises = get_hypothesises(args.input_file)

    idx = 0

    total_bleu_1 = 0
    total_bleu_2 = 0
    total_bleu_3 = 0
    total_bleu_4 = 0

    relation_bleu_1 = defaultdict(lambda: defaultdict(int))

    count = 0

    for head_relation, references, hypothesis in zip(heads_relations, references_list, hypothesises):
        bleu_1 = sentence_bleu(references, hypothesis, weights=[1.0])
        bleu_2 = sentence_bleu(references, hypothesis, weights=[0.5, 0.5])
        bleu_3 = sentence_bleu(references, hypothesis, weights=[0.34, 0.33, 0.33])
        bleu_4 = sentence_bleu(references, hypothesis)

        result = {
            'generation': postprocess(hypothesis),
            'references': [postprocess(reference) for reference in references],
            'input': head_relation
        }
        if (not args.exclude_none and references!=['none']) or hypothesis != 'none':
            # if hypothesis == 'none' and bleu_1==1.0:
            #     #print(f"bleu scores: {bleu_1} | {bleu_2} | {bleu_3} | {bleu_4}")
            #     print(f"gen: {hypothesis}, ref: {references}")

            total_bleu_1 += bleu_1
            total_bleu_2 += bleu_2
            total_bleu_3 += bleu_3
            total_bleu_4 += bleu_4

            relation_bleu_1[head_relation["relation"]]["total"] += bleu_1
            relation_bleu_1[head_relation["relation"]]["count"] += 1

            count += 1

        outfile.write(json.dumps(result) + "\n")
    #print('gens non-none', count)
    print("totoal gens:", count)
    outfile_scores = open(os.path.join(os.path.dirname(input_file), os.path.basename(input_file).split('.')[0] + "_scores.jsonl"), 'w')

    summary = {
        'bleu1': total_bleu_1 / count,
        'bleu2': total_bleu_2 / count,
        'bleu3': total_bleu_3 / count,
        'bleu4': total_bleu_4 / count
    }

    for relation in relation_bleu_1:
        summary[relation] = relation_bleu_1[relation]["total"] / relation_bleu_1[relation]["count"]
        
    outfile_scores.write(json.dumps(summary) + "\n")
    excel_str = ""
    for key in summary:
        excel_str += str(key) + '\t'
    outfile_scores.write(excel_str.strip())
    outfile_scores.write("\n")
    excel_str = ""
    for key in summary:
        excel_str += str(summary[key]) + '\t'

    outfile_scores.write(excel_str.strip())
    print(f"Saved gens in {outfile_path}")
    return(os.path.abspath(outfile_path))


def get2(l):
    return list(zip(*l))[1]


def read_csv(input_file, quotechar='"', delimiter=",", skip_header=False):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_ALL, skipinitialspace=True)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        if skip_header:
            lines = lines[1:]
        return lines


def write_tsv(output_file, data, header=False):
    keys = list(data[0].keys())
    with open(output_file, 'w') as f:
        w = csv.DictWriter(f, keys, delimiter='\t', lineterminator='\n')
        if header:
            w.writeheader()
        for r in data:
            entry = {k: r[k] for k in keys}
            w.writerow(entry)


def write_array2tsv(output_file, data, header=False):
    keys = range(len(data[0]))
    with open(output_file, 'w') as f:
        w = csv.DictWriter(f, keys, delimiter='\t', lineterminator='\n')
        if header:
            w.writeheader()
        for r in data:
            entry = {k: r[k] for k in keys}
            w.writerow(entry)


def write_csv(filename, data, fieldnames):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            formatted_d = {}
            for key, val in d.items():
                formatted_d[key] = json.dumps(val)
            writer.writerow(formatted_d)


def read_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_items(output_file, items):
    dir_name = os.path.dirname(output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()


def write_jsonl(f, d):
    write_items(f, [json.dumps(r) for r in d])


def count_relation(d):
    relation_count = {}
    prefix_count = {}
    head_count = {}
    for l in d:
        r = l[1]
        if r not in relation_count.keys():
            relation_count[r] = 0
        relation_count[r] += 1

        prefix = l[0]+l[1]
        if prefix not in prefix_count.keys():
            prefix_count[prefix] = 0
        prefix_count[prefix] += 1

        head = l[0]
        if head not in head_count.keys():
            head_count[head] = 0
        head_count[head] += 1

    sorted_relation_count = dict(sorted(relation_count.items(), key=operator.itemgetter(1), reverse=True))
    sorted_prefix_count = dict(sorted(prefix_count.items(), key=operator.itemgetter(1), reverse=True))
    sorted_head_count = dict(sorted(head_count.items(), key=operator.itemgetter(1), reverse=True))

    print("Relations:")
    for r in sorted_relation_count.keys():
        print(r, sorted_relation_count[r])

    print("\nPrefixes:")
    print("uniq prefixes: ", len(sorted_prefix_count.keys()))
    i = 0
    for r in sorted_prefix_count.keys():
        print(r, sorted_prefix_count[r])
        i += 1
        if i > 20:
            break

    print("\nHeads:")
    i = 0
    for r in sorted_head_count.keys():
        print(r, sorted_head_count[r])
        i += 1
        if i > 20:
            break


def get_head_set(d):
    return set([l[0] for l in d])


def head_based_split(data, dev_size, test_size, head_size_threshold=500, dev_heads=[], test_heads=[]):
    """
    :param data: the tuples to split according to the heads, where the head is the first element of each tuple
    :param dev_size: target size of the dev set
    :param test_size: target size of the test set
    :param head_size_threshold: Maximum number of tuples a head can be involved in,
    in order to be considered for the dev/test set'
    :param dev_heads: heads that are forced to belong to the dev set
    :param test_heads: heads that are forced to belong to the test set
    :return:
    """
    head_count = {}
    for l in data:
        head = l[0]
        if head not in head_count.keys():
            head_count[head] = 0
        head_count[head] += 1

    remaining_heads = dict(head_count)

    test_selected_heads = {}
    test_head_total_count = 0

    for h in test_heads:
        if h in remaining_heads:
            c = remaining_heads[h]
            test_selected_heads[h] = c
            test_head_total_count += c
            remaining_heads.pop(h)

    while test_head_total_count < test_size:
        h = random.sample(remaining_heads.keys(), 1)[0]
        c = remaining_heads[h]
        if c < head_size_threshold:
            test_selected_heads[h] = c
            test_head_total_count += c
            remaining_heads.pop(h)

    test = [l for l in data if l[0] in test_selected_heads.keys()]

    dev_selected_heads = {}
    dev_head_total_count = 0

    for h in dev_heads:
        if h in remaining_heads:
            c = remaining_heads[h]
            dev_selected_heads[h] = c
            dev_head_total_count += c
            remaining_heads.pop(h)

    while dev_head_total_count < dev_size:
        h = random.sample(remaining_heads.keys(), 1)[0]
        c = remaining_heads[h]
        if c < head_size_threshold:
            dev_selected_heads[h] = c
            dev_head_total_count += c
            remaining_heads.pop(h)

    dev = [l for l in data if l[0] in dev_selected_heads.keys()]

    dev_test_heads = set(list(dev_selected_heads.keys()) + list(test_selected_heads.keys()))
    train = [l for l in data if l[0] not in dev_test_heads]

    return train, dev, test


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]