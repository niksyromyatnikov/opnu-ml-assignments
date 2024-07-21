import string
from typing import Dict, List, Tuple


# Знаки пунктуації
punct = set(string.punctuation)

# Морфологічні правила визначення тегів для слів, що відсутні у словнику
nn_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
vb_suffix = ["ate", "ify", "ise", "ize"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adv_suffix = ["ward", "wards", "wise"]


def get_word_and_tag(line: str, vocab: Dict[str, int]) -> Tuple[str, str]:
    word_and_tag = line.split() 
    
    if not word_and_tag:
        word = "--n--"
        tag = "--s--"
        return word, tag
    
    word, tag = word_and_tag

    if word not in vocab: 
        # Обробка невідомих слів
        word = resolve_unknown(word)
    
    return word, tag


def preprocess(vocab: Dict[str, int], data_fp: str) -> Tuple[List[str], List[str]]:
    """
    Обробити вхідний датасет data_fp зі словами та тегами.
    Повернути оригінальний набір, а також модифікований для передбачення (тільки слова)
    """
    orig = []
    prep = []

    # Завантаження набору даних
    with open(data_fp, "r") as data_file:

        for word in data_file:

            # Кінець речення
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Обробка невідомих слів
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = resolve_unknown(word)
                prep.append(word)
                continue

            else:
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep


def resolve_unknown(tok: str) -> str:
    """Ідентифікація невідомих слів"""
    # Цифри
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"

    # Пунктуація
    elif any(char in punct for char in tok):
        return "--unk_punct--"

    # Верхній регістр
    elif any(char.isupper() for char in tok):
        return "--unk_upper--"

    # Іменники
    elif any(tok.endswith(suffix) for suffix in nn_suffix):
        return "--unk_noun--"

    # Дієслова
    elif any(tok.endswith(suffix) for suffix in vb_suffix):
        return "--unk_verb--"

    # Прикметники
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Прислівники
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    return "--unk--"