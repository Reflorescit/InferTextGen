import sys
import os
import pickle

from OpenPrompt.openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate, PrefixTuningTemplate


import logging
logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        ]
    )


def manual_template_str():
    head = "{'placeholder': 'text_a'}"
    art_h = "{'meta': 'art_h'}"
    art_t = "{'meta': 'art_t'}"
    vp = "{'meta': 'vp'}"
    mask = "{'mask'}"
    re =  {
        "AtLocation": f"You are likely to find {art_h} {head} in {art_t} {mask}",
        "CapableOf": f"{head} can {mask}",
        "Causes": f"Sometimes {head} causes {mask}",
        "Desires": f"{art_h} {head} desires {mask}",
        "HasProperty": f"{head} is {mask}",
        "HasSubEvent": f"While {vp}, you would {mask}",
        "HinderedBy": f"{head}. This would not happen if {mask}",
        "MadeUpOf": f"{art_h} {head} contains {mask}",
        "NotDesires": f"{art_h} {head} does not desire {mask}",
        "ObjectUse": f"{art_h} {head} can be used for {mask}",
        "isAfter": f"{head}. Before that, {mask}",
        "isBefore": f"{head}. After that, {mask}",
        "isFilledBy": f"{head} is filled by {mask}",
        "oEffect": f"{head}. The effect on others will be {mask}",
        "oReact": f"{head}. As a result, others feel {mask}",
        "oWant": f"{head}. After, others will want to {mask}",
        "xAttr": f"{head}. PersonX is {mask}",
        "xEffect": f"{head}. The effect on PersonX will be {mask}",
        "xIntent": f"{head}. PersonX did this to {mask}",
        "xNeed": f"{head}. Before, PersonX needs to {mask}",
        "xReact": f"{head}. PersonX will be {mask}",
        "xReason": f"{head}. PersonX did this because {mask}",
        "xWant": f"{head}. After, PersonX will want to {mask}"
    }
    return re

def soft_template_str():
    head = "{'placeholder': 'text_a'}"
    mask = "{'mask'}"
    soft = "{'soft'}"
    soften = lambda text : f"{{'soft': '{text}'}}"
    n=100
    re =  {
        "AtLocation": f"{soften('You are likely to find a')} {head} {soften('in a')} {mask}",
        "CapableOf": f"{soft*n} {head} {soften('is able to')} {mask}",
        "Causes": f"{soft*n} {soften('Sometimes')} {head} {{'soft': 'results in'}} {mask}",
        "Desires": f"{soft*n} {head} {soften('desires')} {mask}",
        "HasProperty": f"{soft*n} {head} {soften('has the property of')} {mask}",
        "HasSubEvent": f"{soft*n} {soften('While')} {head}, {soften('you would')} {mask}",
        "HinderedBy": f"{soft*n} {head} {soften('. This would not happen if')} {mask}",
        "MadeUpOf": f"{soft*n} {soften('a')} {head} {soften('is made up of')} {mask}",
        "NotDesires": f"{soft*n} {soften('a')} {head} {soften('does not desire')} {mask}",
        "ObjectUse": f"{soft*n} {soften('a')} {head} {soften('can be used for')} {mask}",
        "isAfter": f"{soft*n} {head} {soften('. Before that,')} {mask}",
        "isBefore": f"{soft*n} {head} {soften('. After that,')} {mask}",
        "isFilledBy": f"{soft*n} {head} {soften('is filled by')} {mask}",
        "oEffect": f"{soft*n} {head} {soften('. The effect on others will be')} {mask}",
        "oReact": f"{soft*n} {head} {soften('. As a result, others feel')} {mask}",
        "oWant": f"{soft*n} {head} {soften('. After, others will want to')} {mask}",
        "xAttr": f"{soft*n} {head} {soften('. PersonX is')} {mask}",
        "xEffect": f"{soft*n} {head} {soften('. The effect on PersonX will be')} {mask}",
        "xIntent": f"{soft*n} {head} {soften('. PersonX did this to')} {mask}",
        "xNeed": f"{soft*n} {head} {soften('. Before, PersonX needs to')} {mask}",
        "xReact": f"{soft*n} {head} {soften('. PersonX will be')} {mask}",
        "xReason": f"{soft*n} {head} {soften('. PersonX did this because')} {mask}",
        "xWant": f"{soft*n} {head}{soften('. After, PersonX will want to')} {mask}"
    }
    return re

def promptTuning_template_str(soft_len):
    head = "{'placeholder': 'text_a'}"
    mask = "{'mask'}"
    soft = "{'soft'}"
    soften = lambda text : f"{{'soft': '{text}'}}"
    n=soft_len
    re =  {
        "AtLocation": f"{soften('location')*n} {head} {{'special': '<eos>'}} {mask}",
        "CapableOf": f"{soften('capable')*n} {head} {{'special': '<eos>'}} {mask}",
        "Causes": f"{soften('cause')*n} {head} {{'special': '<eos>'}} {mask}",
        "Desires": f"{soften('desire')*n} {head} {{'special': '<eos>'}} {mask}",
        "HasProperty": f"{soften('property')*n} {head} {{'special': '<eos>'}} {mask}",
        "HasSubEvent": f"{soften('subEvent')*n} {head} {{'special': '<eos>'}} {mask}",
        "HinderedBy": f"{soften('hindered')*n} {head} {{'special': '<eos>'}} {mask}",
        "MadeUpOf": f"{soften('consist')*n} {head} {{'special': '<eos>'}} {mask}",
        "NotDesires": f"{soften('unwilling')*n} {head} {{'special': '<eos>'}} {mask}",
        "ObjectUse": f"{soften('use')*n} {head} {{'special': '<eos>'}} {mask}",
        "isAfter": f"{soften('after')*n} {head} {{'special': '<eos>'}} {mask}",
        "isBefore": f"{soften('before')*n} {head} {{'special': '<eos>'}} {mask}",
        "isFilledBy": f"{soften('filled')*n} {head} {{'special': '<eos>'}} {mask}",
        "oEffect": f"{soften('effect')*n} {head} {{'special': '<eos>'}} {mask}",
        "oReact": f"{soften('reaction')*n} {head} {{'special': '<eos>'}} {mask}",
        "oWant": f"{soften('want')*n} {head} {{'special': '<eos>'}} {mask}",
        "xAttr": f"{soften('attribute')*n} {head} {{'special': '<eos>'}} {mask}",
        "xEffect": f"{soften('effect')*n} {head} {{'special': '<eos>'}} {mask}",
        "xIntent": f"{soften('intention')*n} {head} {{'special': '<eos>'}} {mask}",
        "xNeed": f"{soften('need')*n} {head} {{'special': '<eos>'}} {mask}",
        "xReact": f"{soften('reaction')*n} {head} {{'special': '<eos>'}} {mask}",
        "xReason": f"{soften('because')*n} {head} {{'special': '<eos>'}} {mask}",
        "xWant": f"{soften('want')*n} {head} {{'special': '<eos>'}} {mask}",
    }
    return re

def mix_template_str():
    head = "{'placeholder': 'text_a'}"
    mask = "{'mask'}"
    soft = "{'soft'}"
    soften = lambda text : f"{{'soft': '{text}'}}"
    n=3
    re =  {
        "AtLocation": f"{soften('You are likely to find a')} {head} {soften('in a')} {mask}",
        "CapableOf": f"{head} {soften('is able to')} {mask}",
        "Causes": f"{soften('Sometimes')} {head} {{'soft': 'results in'}} {mask}",
        "Desires": f"{head} {soften('desires')} {mask}",
        "HasProperty": f"{head} {soften('has the property of')} {mask}",
        "HasSubEvent": f"{soften('While')} {head}, {soften('you would')} {mask}",
        "HinderedBy": f"{head} {soften('. This would not happen if')} {mask}",
        "MadeUpOf": f"{soften('a')} {head} {soften('is made up of')} {mask}",
        "NotDesires": f"{soften('a')} {head} {soften('does not desire')} {mask}",
        "ObjectUse": f"{soften('a')} {head} {soften('can be used for')} {mask}",
        "isAfter": f"{head} {soften('. Before that,')} {mask}",
        "isBefore": f"{head} {soften('. After that,')} {mask}",
        "isFilledBy": f"{head} {soften('is filled by')} {mask}",
        "oEffect": f"{head} {soften('. The effect on others will be')} {mask}",
        "oReact": f"{head} {soften('. As a result, others feel')} {mask}",
        "oWant": f"{head} {soften('. After, others will want to')} {mask}",
        "xAttr": f"{head} {soften('. PersonX is')} {mask}",
        "xEffect": f"{head} {soften('. The effect on PersonX will be')} {mask}",
        "xIntent": f"{head} {soften('. PersonX did this to')} {mask}",
        "xNeed": f"{head} {soften('. Before, PersonX needs to')} {mask}",
        "xReact": f"{head} {soften('. PersonX will be')} {mask}",
        "xReason": f"{head} {soften('. PersonX did this because')} {mask}",
        "xWant": f"{head}{soften('. After, PersonX will want to')} {mask}"
    }
    return re


def init_template(method, plm, tokenizer, soft_len=0, rel=None):
    def init_from_vocab():
        """initialize soft tokens from most frequent word-pieces in plm's vocab"""
        bpe_lst = [tu[0] for tu in sorted(tokenizer.bpe_ranks.items(), key=lambda kv:kv[1])[:soft_len]]
        wps_lst = ["".join(bpe).replace("Ġ", " ").replace("'", "\'") for bpe in bpe_lst]
        soften = lambda text : f'{{"soft": "{text}"}}'
        soft_tkn_template = ""
        for wps in wps_lst:
            soft_tkn_template += soften(wps)
        return soft_tkn_template

    soft = "{'soft': ''}"
    prompt = "{'meta':'prompt'}"
    mask = "{'mask'}"
    if method == "manual":
        text = manual_template_str()[rel]
        logger.info("init manual template: {}".format(text))
        return ManualTemplate(tokenizer=tokenizer, text=text)
    elif method == 'soft':
        # text = promptTuning_template_str(soft_len)[rel]
        soft_tkn_template = init_from_vocab()
        text = f"{soft_tkn_template} {prompt} {mask}"
        logger.info("init prompt-tuning template: {}".format(text))
        return MixedTemplate(model=plm, tokenizer=tokenizer, text=text)
    elif method == "prefixTuning":
        logger.info("init prfix-tuning template with {} soft tokens".format(soft_len))
        text = f"{prompt} {mask}"
        return PrefixTuningTemplate(model=plm, tokenizer=tokenizer, num_token=soft_len, text = text)


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





if __name__ == "__main__":
    re = soft_template_str()
    for k in re.keys():
        print("{}: {}".format(k, re[k]))

