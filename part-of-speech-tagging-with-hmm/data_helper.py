from typing import Tuple
from pos_helper import preprocess


def load_data(
    train_set_path: str = "data/train_pos.txt",
    full_test_set_path: str = "data/test_pos.txt",
    model_vocab_path: str = "data/vocab.txt",
    pred_test_set_path: str = "data/test_predict.txt"
) -> Tuple[list, list, dict, list]:
    # завантаження тренувального набору
    with open(train_set_path, 'r') as f:
        train_set = f.readlines()

    print(f"Перші 20 рядків тренувального набору:\n{train_set[0:20]}\n")

    # завантаження словника
    with open(model_vocab_path, 'r') as f:
        voc_l = f.read().split('\n')

    print(f"\nПерші 20 елементів з файлу словника: {voc_l[0:20]}\nОстанні 20 елементів з файлу словника: {voc_l[-20:]}\n")

    vocab = {} 

    # Встановлення індексів для слів зі словника
    for i, word in enumerate(sorted(voc_l)): 
        vocab[word] = i       
    
    print("Перші 20 елементів словника:\nКлюч: Значення")
    cnt = 0
    for k, v in vocab.items():
        print(f"{k}: {v}")
        cnt += 1
        if cnt > 20:
            break

    # завантаження тестового набору
    with open(full_test_set_path, 'r') as f:
        full_test_set = f.readlines()

    print("\n\nПерші 20 рядків тестового набору:")
    print(full_test_set[0:20])

    # завантаження та підготовка набору для передбачення
    _, pred_test_set = preprocess(vocab, pred_test_set_path)     

    print(f'\n\nПерші 20 елементів набору для передбачення:\n{pred_test_set[0:20]}')

    return train_set, full_test_set, vocab, pred_test_set