# AI Spam Message Classifier

A self-contained text classifier that sorts SMS-style messages into **spam**
or **ham** (not spam). Everything — the labeled dataset, preprocessing, model
training, evaluation, and the CLI — lives in one file,
`spam_classifier_all_in_one.py`.

## How it works

```
message -> clean_text() -> TF-IDF (1-2 grams) -> classifier -> spam | ham
```

1. **Clean** — lowercase, strip URLs, digits, and punctuation, collapse whitespace.
2. **Vectorize** — TF-IDF over unigrams and bigrams, English stop words removed.
3. **Train two models** — Multinomial Naive Bayes and Logistic Regression.
4. **Keep the better one** — whichever scores higher F1 on the held-out split.
5. **Save** — model and vectorizer are written to `.joblib` files so later runs
   skip training.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train, evaluate, save, and run a short demo
python spam_classifier_all_in_one.py

# Classify one message using the saved model
python spam_classifier_all_in_one.py --classify "Free prize! Click now"

# Type messages interactively (Ctrl+C to quit)
python spam_classifier_all_in_one.py --interactive

# Force retraining even if a saved model exists
python spam_classifier_all_in_one.py --classify "..." --retrain
```

Or import it:

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

Both models, for the best model:

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
matrix**, meaning they make the same prediction on all 21 test messages.
`train_and_evaluate()` reports Naive Bayes as "best" only because it compares
with `f1 > best_score` and iterates NB first; on a tie the first model wins.
That is a coin toss presented as a result, not evidence that Naive Bayes is
the better choice here.

Two very different algorithms agreeing perfectly is a symptom of the data,
not a sign of a strong model: the spam and ham messages are separable by
obvious keywords (`free`, `click`, `winner`, `urgent`), so almost any
classifier lands in the same place. Both miss the same 2 spam messages.

### Read those numbers with suspicion

**The dataset is 81 messages** (40 spam, 41 ham), embedded directly in the
script as `DATASET_CSV`. A 25% test split is **21 messages**, so each one is
worth ~4.8 percentage points of accuracy and the confusion matrix above is
counting single-digit numbers. The metrics describe this toy dataset, not
real-world performance, and they will swing noticeably if you change
`random_state`.

The messages are also *written* spam rather than *collected* spam — clean,
unambiguous, and free of the obfuscation real spam uses (`Fr33`, unicode
lookalikes, image-only payloads). Expect a real corpus to be much harder.

To get numbers that mean something, swap in a real dataset such as the
UCI SMS Spam Collection (~5,570 messages), keeping the `label,text` format,
and use cross-validation rather than a single split.

## Extending it

- **More data** — replace `DATASET_CSV`, or change `load_data()` to read an
  external CSV.
- **Other models** — `sklearn.svm.LinearSVC` and
  `sklearn.neural_network.MLPClassifier` are drop-in swaps. Note `LinearSVC`
  has no `predict_proba`, and `predict_message()` degrades to no confidence
  score rather than failing.
- **Serve it** — wrap `predict_message()` in a Flask or FastAPI endpoint.

## Note on the saved model

`spam_model.joblib` and `vectorizer.joblib` are git-ignored. Pickled
scikit-learn objects are tied to the version that wrote them and frequently
refuse to load under a different release, so committing them would ship a
file that breaks elsewhere. Training takes under a second — just rerun.
