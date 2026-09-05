# AI Spam Message Classifier

A self-contained machine learning project that classifies text messages as
**spam** or **ham** (not spam) using scikit-learn.

There are three ways in, all driving the same TF-IDF pipeline:

| | |
|---|---|
| **Multi-file scripts** | `dataset.csv` + `train_spam_classifier.py` + `classify.py` — train and classify as separate steps |
| **Single file** | `spam_classifier_all_in_one.py` — the same pipeline with the dataset embedded, so it runs anywhere with no other files |
| **Web UI** | `app.py` — browser interface that also shows *which words* drove each verdict |

## Files

| File | Purpose |
|---|---|
| `dataset.csv` | 81 labeled sample messages (spam/ham) used for training |
| `train_spam_classifier.py` | Cleans data, extracts TF-IDF features, trains and compares Naive Bayes vs Logistic Regression, prints metrics, saves the best model |
| `classify.py` | Command-line tool to classify new messages using the saved model |
| `spam_classifier_all_in_one.py` | All of the above in one file, with `dataset.csv` inlined as `DATASET_CSV` |
| `app.py` | Flask web UI + per-word explanations |
| `templates/`, `static/` | The page, styling, and browser wiring |
| `spam_model.joblib`, `vectorizer.joblib` | Saved model + vectorizer, created by training (git-ignored) |

## How it works

```
message -> clean_text() -> TF-IDF (1-2 grams) -> classifier -> spam | ham
```

1. **Preprocessing** — lowercase, strip URLs, digits, and punctuation.
2. **Feature extraction** — `TfidfVectorizer` with unigrams and bigrams turns
   each message into a numeric vector, down-weighting common English words.
3. **Model** — `MultinomialNB` is compared against `LogisticRegression`;
   whichever scores higher on F1 is kept.
4. **Evaluation** — accuracy, precision, recall, F1, and a confusion matrix
   on a held-out test split.
5. **Inference** — new messages are cleaned and vectorized the same way, then
   run through the saved model.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Multi-file scripts

```bash
python train_spam_classifier.py                                   # train + evaluate
python classify.py "Congratulations! You've won a free prize"     # classify one
python classify.py                                                # interactive
```

### Single file

```bash
python spam_classifier_all_in_one.py                              # train + demo
python spam_classifier_all_in_one.py --classify "Free prize!"
python spam_classifier_all_in_one.py --interactive
python spam_classifier_all_in_one.py --classify "..." --retrain
```

### Web UI

```bash
python app.py
# then open http://127.0.0.1:5002
```

Paste a message and it shows the verdict, the confidence, and — the part the
CLI cannot give you — **which words drove the decision**, colour-coded by
whether each pushed towards spam or ham. Paste several lines and hit
*Classify each line* for a batch table.

Both classifiers reduce to a linear score over TF-IDF features, so a token's
contribution is its TF-IDF value times the model's weight for it. The two
models store that weight in different places (`coef_` for Logistic
Regression, the difference between the two rows of `feature_log_prob_` for
Naive Bayes), and `token_weights()` in `app.py` handles both.

It binds to `127.0.0.1`. The endpoints are unauthenticated, and messages
pasted in for testing are the sort of thing you would rather not expose.

| Endpoint | Purpose |
|---|---|
| `GET /` | the page |
| `POST /api/classify` | `{message}` → verdict, confidence, token weights |
| `POST /api/batch` | `{messages: [...]}` → one verdict per line (cap 200) |
| `POST /api/retrain` | retrain from the embedded dataset and reload |
| `GET /api/model` | which model is loaded, and its scores |

### As a library

```python
from spam_classifier_all_in_one import load_artifacts, predict_message

model, vectorizer = load_artifacts()
print(predict_message("Congratulations, you won!", model, vectorizer))
# spam (confidence: 77.23%)
```

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.905 | 1.00 | 0.80 | 0.889 |
| Logistic Regression | 0.905 | 1.00 | 0.80 | 0.889 |

Confusion matrix (identical for both):

```
          predicted
          ham  spam
ham  [[   11     0  ]
spam  [    2     8  ]]
```

No false positives — no real message was called spam — at the cost of missing
2 of 10 spam messages. For a spam filter that is usually the right side to err
on: a missed spam is an annoyance, a misfiled real message is a lost message.

### The two models are tied, and that is the finding

They do not merely score the same — they produce the **identical confusion
matrix**, meaning they make the same prediction on all 21 test messages. The
training script reports Naive Bayes as "best" only because it compares with
`f1 > best_score` and iterates NB first; on a tie the first model wins. That
is a coin toss presented as a result, not evidence that Naive Bayes is the
better choice here.

Two very different algorithms agreeing perfectly is a symptom of the data,
not a sign of a strong model: the spam and ham messages are separable by
obvious keywords (`free`, `click`, `winner`, `urgent`), so almost any
classifier lands in the same place. Both miss the same 2 spam messages.

### Read those numbers with suspicion

**The dataset is 81 messages** (40 spam, 41 ham). A 25% test split is
**21 messages**, so each one is worth ~4.8 percentage points of accuracy and
the confusion matrix above is counting single-digit numbers. The metrics
describe this toy dataset, not real-world performance, and they will swing
noticeably if you change `random_state`.

The messages are also *written* spam rather than *collected* spam — clean,
unambiguous, and free of the obfuscation real spam uses (`Fr33`, unicode
lookalikes, image-only payloads). Expect a real corpus to be much harder.

To get numbers that mean something, swap in a real dataset such as the
UCI SMS Spam Collection (~5,570 messages), keeping the `label,text` format,
and use cross-validation rather than a single split.

## Extending it

- **More data** — replace `dataset.csv` (or `DATASET_CSV` in the all-in-one
  version), keeping the `label,text` format.
- **Other models** — `sklearn.svm.LinearSVC` and
  `sklearn.neural_network.MLPClassifier` are drop-in swaps. Note `LinearSVC`
  has no `predict_proba`, and the confidence score degrades to absent rather
  than failing.
- **Serve it** — `app.py` is the reference for wrapping the model in HTTP.

## Note on the saved model

`spam_model.joblib` and `vectorizer.joblib` are git-ignored. Pickled
scikit-learn objects are tied to the version that wrote them and frequently
refuse to load under a different release, so committing them would ship a
file that breaks elsewhere. Training takes under a second — just rerun.
