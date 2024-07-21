from typing import Iterable


def calculate_accuracy(pred_y: Iterable[str], true_y: Iterable[str]) -> float:
    """
    pred_y: список з передбаченими тегами (частинами мови) 
    true_y: список рядків, де кожний рядок представлено у вигляді "word\ttag"
    """
    correct_preds = 0
    total_preds = 0
    
    for prediction, true_y in zip(pred_y, true_y):
        # поділ рядків на слово та тег
        word_and_tag = true_y.split()

        if len(word_and_tag) != 2:
            continue 
        
        # перевірка співпадіння передбаченого і правильного тегів
        correct_preds += prediction == word_and_tag[-1]
        total_preds += 1

    return correct_preds / total_preds