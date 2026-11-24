# Natural Language Processing Labs

[Українською](README.uk.md)

Three hands-on NLP assignments in Python Jupyter notebooks, with explanations and exercises in Ukrainian.

| Notebook | What you build | How you evaluate it |
| --- | --- | --- |
| [1. Part-of-speech tagging](part-of-speech-tagging-with-hmm/pos_hmm_task.ipynb) | A Viterbi decoder that assigns a grammatical tag to each word in English news text | Token-level tagging accuracy |
| [2. Ukrainian news headline generation](2026-news-title-generation-gpt2-lora/news-title-generation-gpt2-lora.ipynb) | A GPT-2 headline generator fine-tuned with LoRA, compared with a prompted Qwen model | BERTScore against the original headlines |
| [3. Open-domain question answering](2026-open-domain-question-answering/open-domain-question-answering.ipynb) | A question-answering system with FAISS passage retrieval and an mDeBERTa extractive reader | Exact match and token F1 on supplied and retrieved passages |

## 1. Part-of-speech tagging with a hidden Markov model

The word *model* can be a noun or a verb. A hidden Markov model (HMM) uses both the word and its context to choose a tag. The lab supplies the corpus and the code for estimating the model. The programming assignment is to implement the decoder that finds the most likely sequence of tags.

1. **Prepare the Wall Street Journal corpus.** Load the tagged training text, vocabulary, and untagged test text. Inspect how the preprocessing code represents sentence boundaries and groups unfamiliar words using clues such as digits, capitalization, and suffixes.
2. **Estimate transition and emission probabilities.** The helpers count individual tags, adjacent tag pairs, and tag–word pairs. Additive smoothing turns the counts into two matrices. The transition matrix describes which tag tends to follow another. The emission matrix describes which words each tag tends to produce. Inspecting entries for words such as *model* shows why both matrices matter.
3. **Implement Viterbi decoding.** Complete `initialize()`, `forward()`, and `backward()`. Initialize the scores at the start of the sequence, track the best log-probability and predecessor for each possible tag at each position, then follow the stored predecessors to recover the tag sequence.
4. **Evaluate the predictions.** Inspect words alongside their predicted and reference tags. Compute token accuracy on the tagged test set as the fraction of scored words assigned the correct tag. Sentence-boundary rows are excluded from the denominator.

**Run locally.** Open the notebook in Jupyter from the lab directory so Python can find the helper modules and `data/`. The corpus is included. Skip the optional archive download and run the dependency cell to install NumPy and pandas.

## 2. Ukrainian news headline generation with GPT-2 and LoRA

Given the body of a news article, generate a short Ukrainian headline. The lab compares three ways to approach the task — generation with a base language model, parameter-efficient fine-tuning on article–headline pairs, and prompting a larger model through an API.

1. **Prepare the news data.** Load the prepared subset of `FIdo-AI/ua-news`, split it into training, validation, and test sets, and filter out texts repeated across splits. The notebook uses 3,000 validation examples and 500 test articles. Each record contains the original headline. During generation, the prompt ends at `Назва:` and the headline is held back for evaluation.
2. **Establish a baseline.** Generate headlines with `benjamin/gpt2-large-wechsel-ukrainian` before fine-tuning and inspect whether they reflect the articles. GPT-2 generates one token at a time. Sampling temperature controls how strongly it favors the most likely next tokens.
3. **Adapt GPT-2 with LoRA.** Freeze the base weights, attach rank-16 low-rank adapters with Hugging Face PEFT, and inspect how many parameters remain trainable. The assignment variants specify the training budget, learning-rate warmup, evaluation interval, and checkpoint schedule. Generate headlines with the adapted model at the assigned temperature.
4. **Compare with a prompted model.** Send the same test articles to Qwen3.8-27B through the Groq API with a Ukrainian instruction to write a headline. The extension asks students to improve these generations. The Groq API key is needed for this experiment.
5. **Score both sets of headlines.** BERTScore uses `microsoft/mdeberta-v3-base` to compare contextual token representations in each generated headline and its reference. Precision measures how well the generated tokens match the reference, recall measures coverage in the other direction, and F1 combines the two. The notebook averages each metric over the test set.

| Saved notebook results on 500 articles | BERTScore precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| GPT-2 with LoRA | 0.622 | 0.700 | 0.657 |
| Qwen3-32B via Groq | 0.689 | 0.681 | 0.684 |

**Run on Kaggle or Google Colab.** Select a GPU, and attach the prepared dataset. To use the published model, complete the setup through section 2.2, skip section 2.3, and select the Hugging Face loading cell in section 3 for `NLPForUA/gpt2-large-uk-title-generation`. If you trained your own adapter, select the local checkpoint cell instead and point it to your checkpoint. [Prepared Kaggle dataset](https://www.kaggle.com/datasets/niksyromyatnikov/fido-dataset)

## 3. Open-domain question answering in Ukrainian

An extractive reader can answer a question when given the right passage. In open-domain question answering (ODQA), the system must first find that passage in a collection. This lab evaluates both settings on the same test questions to show how the scores change when retrieval is part of the task.

1. **Prepare Ukrainian SQuAD 2.0 data.** Convert records from `FIdo-AI/ua-squad` into questions, passages, and character offsets for answer spans. Keep questions annotated as unanswerable. Split the unique passages 80/10/10 across training, validation, and test sets so questions about the same passage stay together. Check for overlapping passages and duplicate question–passage pairs.
2. **Build the extractive reader.** Fine-tune `microsoft/mdeberta-v3-base` to predict the start and end tokens of an answer. During training, overlapping token windows handle long passages. At inference, the reader uses character offsets to return the original text span and can return an empty answer when the no-answer score is higher.
3. **Build the passage retriever.** Encode the collection’s unique passages with `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` and index the vectors in FAISS. For each question, retrieve nearby passages, run the reader on each one, and use its scores to select an answer. The end-to-end evaluation retrieves five passages per question.
4. **Measure the effect of retrieval.** Score the reader first with the reference passage and then with retrieved passages. SQuAD exact match checks whether the normalized answer matches the reference. Token F1 gives partial credit for overlap. The notebook also reports scores for questions with and without an annotated answer.
5. **Implement top-N evaluation.** Extend the evaluator to give credit when a correct answer appears among the highest-ranked candidates. Compare the result with the top-ranked answer alone. Assignment variants use N = 3, 5, 7, or 10.

| Saved notebook results | Exact match (0–100) | Token F1 (0–100) |
| --- | ---: | ---: |
| Reader given the reference passage | 58.75 | 72.24 |
| Retriever and reader, five passages per question | 38.96 | 47.84 |

**Run on Kaggle or Google Colab.** Select a GPU runtime with internet access and follow the dependency installation and restart steps. Set `TRAIN_MODEL = False` in the reader setup to load `NLPForUA/mdeberta-v3-ua-squad-reader`. Leave it set to `True` to fine-tune the base model.

## Credits

Labs 2 and 3 are original assignments developed for this course. Lab 1 is partly adapted from DeepLearning.AI’s Natural Language Processing Specialization on Coursera. [Specialization overview](https://www.deeplearning.ai/specializations/natural-language-processing/)
