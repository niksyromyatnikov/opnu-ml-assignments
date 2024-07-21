from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from pos_helper import get_word_and_tag


def calculate_dicts(train_set: Iterable[str], vocab: Dict[str, int]) -> Tuple[dict, dict, dict]:
    """
    train_set: тренувальний набір, де кожний рядок містить слово та тег (частину мови)
    vocab: словник, де слово є ключем, а індекс - значенням
    
    emission_counter: словник, де комбінація (тег, слово) є ключем, а число їх збігів - значенням 
    transition_counter: словник, де комбінація (попередній тег, поточний тег) є ключем, а число їх збігів - значенням
    tag_counter: словник, де тег є ключем, а число його появи у наборі - значенням
    """
    
    # ініціалізація словників
    emission_counter = defaultdict(int)
    transition_counter = defaultdict(int)
    tag_counter = defaultdict(int)
    
    # ініціалізація попереднього тегу "prev_tag" початковим станом '--s--'
    prev_tag = '--s--' 
    
    i = 0 
    
    for word_tag in train_set:
        
        i += 1
        if i % 10**5 == 0:
            print(f"оброблено {i} елементів")

        # визначення слова та тегу (частини мови)
        word, tag = get_word_and_tag(word_tag, vocab) 
        
        # Збільшення числа переходів з попереднього тегу у поточний
        transition_counter[(prev_tag, tag)] += 1
        
        # Збільшення числа виходів з поточного тегу у поточне слово
        emission_counter[(tag, word)] += 1

        # Збільшення числа появи поточного тегу
        tag_counter[tag] += 1

        # Встановлення поточного тегу у якості поперднього для наступної ітерації
        prev_tag = tag
        
    return emission_counter, transition_counter, tag_counter


def build_transitions(
    transition_counter: Dict[Tuple[str, str], int],
    tag_counter: Dict[str, int],
    alpha: float = 0.001
) -> List[List[float]]:
    """
    transition_counter: словник, де комбінація (попередній тег, поточний тег) є ключем, а число їх збігів - значенням
    tag_counter: словник, де тег є ключем, а число його появи у наборі - значенням
    alpha: число, що використовується для згладжування
    
    T: матриця розмірністю tags_total x tags_total
    """
    # сортування унікальних тегів
    tags_list = sorted(tag_counter.keys())
    
    # визначення числа тегів
    tags_total = len(tags_list)
    
    # ініціалізація матриці переходів T
    T = np.zeros((tags_total, tags_total))
    
    # для кожного рядка матриці переходів T
    for i in range(tags_total):
        
        # для кожного стовпця матриці переходів T
        for j in range(tags_total):

            # встановлення 0 для переходу (попередній тег, поточний тег)
            count = 0
        
            # визначення попереднього та поточного тегів
            key = (tags_list[i], tags_list[j])

            if transition_counter:
                
                # визначення числа переходів (попередній тег, поточний тег)
                count = transition_counter[key]
                
            # визначення числа появи попереднього тегу
            count_prev_tag = tag_counter[tags_list[i]]
            
            # згладжування
            T[i, j] = (count + alpha) / (count_prev_tag + alpha * tags_total)
    
    return T


def build_emissions(
    emission_counter: Dict[Tuple[str, str], int],
    tag_counter: Dict[str, int],
    vocab: Dict[str, int],
    alpha: float = 0.001
) -> List[List[float]]:
    """
    emission_counter: словник, де комбінація (тег, слово) є ключем, а число їх збігів - значенням 
    tag_counter: словник, де тег є ключем, а число його появи у наборі - значенням
    vocab: словник, де слово є ключем, а індекс - значенням
    alpha: число, що використовується для згладжування
    
    E: матриця розмірністю (tags_total, len(vocab))
    """
    
    # визначення кількості тегів
    tags_total = len(tag_counter)
    
    # сортування унікальних тегів
    tags_list = sorted(tag_counter.keys())
    
    # визначення числа унікальних слів у словнику
    words_total = len(vocab)
    
    # ініціалізація матриці виходів E
    E = np.zeros((tags_total, words_total))
    
    # для кожного рядка матриці виходів E
    for i in range(tags_total):
        
        # для кожного стовпця матриці виходів E
        for j in range(words_total):

            # встановлення 0 для виходу (тег, слово)
            count = 0
                    
            # визначення тегу та слова
            key = (tags_list[i], vocab[j])

            # якщо вихід (POS tag, word) наявний у emission_counter
            if key in emission_counter.keys():
        
                # визначення числа виходів (тег, слово)
                count = emission_counter[key]
                
            # визначення числа появи тегу
            tag_count = tag_counter[tags_list[i]]
                
            # згладжування
            E[i, j] = (count + alpha) / (tag_count + alpha * words_total)

    return E