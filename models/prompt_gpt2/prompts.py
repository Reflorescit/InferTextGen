

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

if __name__ == "__main__":
    re = soft_template_str()
    for k in re.keys():
        print("{}: {}".format(k, re[k]))

    # re =  {
    #     "AtLocation": '{{"soft": You are likely to find}} {} {} in {} {}'.format(art_h, head, art_t, mask),
    #     "CapableOf": '{} can {}'.format(head, mask),
    #     "Causes": "Sometimes {} causes {}".format(head, mask),
    #     "Desires": "{} {} desires {}".format(art_h, head, mask),
    #     "HasProperty": "{} is {}".format(head, mask),
    #     "HasSubEvent": "While {}, you would {}".format(vp, mask),
    #     "HinderedBy": "{}. This would not happen if {}".format(head, mask),
    #     "MadeUpOf": "{} {} contains {}".format(art_h, head, mask),
    #     "NotDesires": "{} {} does not desire {}".format(art_h, head, mask),
    #     "ObjectUse": "{} {} can be used for {}".format(art_h, head, mask),
    #     "isAfter": "{}. Before that, {}".format(head, mask),
    #     "isBefore": "{}. After that, {}".format(head, mask),
    #     "isFilledBy": "{} is filled by {}".format(head, mask),
    #     "oEffect": "{}. The effect on others will be {}".format(head, mask),
    #     "oReact": "{}. As a result, others feel {}".format(head, mask),
    #     "oWant": "{}. After, others will want to {}".format(head, mask),
    #     "xAttr": "{}. PersonX is {}".format(head, mask),
    #     "xEffect": "{}. The effect on PersonX will be {}".format(head, mask),
    #     "xIntent": "{}. PersonX did this to {}".format(head, mask),
    #     "xNeed": "{}. Before, PersonX needs to {}".format(head, mask),
    #     "xReact": "{}. PersonX will be {}".format(head, mask),
    #     "xReason": "{}. PersonX did this because {}".format(head, mask),
    #     "xWant": "{}. After, PersonX will want to {}".format(head, mask)
    # }