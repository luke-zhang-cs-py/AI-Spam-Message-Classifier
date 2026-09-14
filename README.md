# AI Spam Message Classifier

[![CI](https://github.com/luke-zhang-cs-py/AI-Spam-Message-Classifier/actions/workflows/python-package.yml/badge.svg)](https://github.com/luke-zhang-cs-py/AI-Spam-Message-Classifier/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg)](https://www.python.org/)

**[Read the overview →](https://luke-zhang-cs-py.github.io/AI-Spam-Message-Classifier/)**
— the result and the four reasons to read it carefully, how a verdict is
explained, and every bug this thing has had.

A self-contained machine learning project that classifies text messages as
**spam** or **ham** (not spam) using scikit-learn.

There are three ways in, all driving one pipeline that lives in `spamlib.py`:

| | |
|---|---|
| **Multi-file scripts** | `train_spam_classifier.py` + `classify.py` — train and classify as separate steps |
| **Single file** | `spam_classifier_all_in_one.py` — train, report and classify from one command |
| **Web UI** | `app.py` — browser interface that also shows *which words* drove each verdict |

Those three used to be three copies of the pipeline, with a test asserting
the copies agreed. It kept them in step without ever reducing how many there
were, so every change meant three edits and a test to confirm you had made
them all. There is one implementation now, and these are entry points to it.

## Files

| File | Purpose |
|---|---|
| `spamlib.py` | The pipeline: normalising, model comparison, the tuned threshold, artifacts |
| `data/*.csv` | The corpus, in shards — new material is a new file rather than an edit to a growing one |
| `train_spam_classifier.py` | Trains from the corpus, prints the comparison, saves the winner |
| `classify.py` | Classifies new messages using the saved model |
| `spam_classifier_all_in_one.py` | Train, report and classify from one command |
| `app.py` | Flask web UI + per-word explanations |
| `embeddings.py` | The **optional** sentence-transformer backend, off by default |
| `tools/compare_backends.py` | Measures whether that backend is worth installing |
| `templates/`, `static/` | The page, styling, and browser wiring |
| `dataset.csv` | The original 81-message corpus, still read as a fallback by a checkout with no `data/` |
| `spam_model.joblib`, `vectorizer.joblib` | Saved model + vectorizer, created by training (git-ignored) |

## How it works

```
message -> clean_text() -> TF-IDF (1-2 grams) -> classifier -> threshold -> spam | ham
```

1. **Normalise** — fold case, then replace each noisy part with a token for
   its *kind*: `urltoken`, `moneytoken`, `shortcodetoken`, `phonetoken`,
   `allcapstoken`. An earlier version deleted all of them, which is ordinary
   advice for topic classification and close to backwards here — in "text WIN
   to 80086" the short code *is* the evidence, and deleting it leaves "text
   WIN to". Keeping the shape and discarding the specifics moved F1 from
   0.768 to 0.784.
2. **Feature extraction** — `TfidfVectorizer` over unigrams and bigrams,
   sublinear, with no stop-word list: "free", "win" and "call" are the signal
   here rather than noise to strip.
3. **Model** — `MultinomialNB`, `ComplementNB` and `LogisticRegression`
   compared with `cross_val_predict`, so no message is scored by a fold that
   trained on it. One split moves F1 by several points on a corpus this size,
   and choosing a model on that is choosing a split.
4. **Threshold** — the cut with the best F1 whose precision stays at or above
   `PRECISION_FLOOR` (0.90). Predicting at 0.5 leaves the most consequential
   dial untouched: blocking a real message costs the reader something they
   wanted, missing a scam costs them an annoyance. The cut is wrapped into
   the model, so a saved artifact cannot forget the one it was chosen with.
5. **Inference** — new messages go through the same normalising and the same
   cut.

## Setup

```bash
pip install -r requirements.txt
```

The embedding backend is a separate install, and the honest advice is to skip
it — see [the measurement](#the-optional-embedding-backend) below.

```bash
pip install -r requirements-embeddings.txt   # ~2 GB, for +0.003 F1
```

## Usage

### Multi-file scripts

```bash
python train_spam_classifier.py                                   # train + evaluate
python train_spam_classifier.py --data dataset.csv                # one file instead
python classify.py "Congratulations! You've won a free prize"     # classify one
python classify.py                                                # interactive
```

### Single file

```bash
python spam_classifier_all_in_one.py                              # train + demo
python spam_classifier_all_in_one.py --classify "Free prize!"
python spam_classifier_all_in_one.py --interactive
```

`--classify` deliberately takes no backend flag: which features to use is a
property of the artifact on disk, not of the command reading it.

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
| `POST /api/retrain` | retrain from `data/` and reload, keeping whichever backend is loaded |
| `GET /api/model` | which model is loaded, and its scores |

### As a library

```python
from spam_classifier_all_in_one import load_artifacts, predict_message

model, vectorizer = load_artifacts()
print(predict_message("Congratulations, you won!", model, vectorizer))
# spam (confidence: 77.23%)
```

## Results

412 messages (233 ham, 179 spam), scored **out of fold** — every probability
comes from a fold that did not train on the message it scores, so all 412 are
held-out data exactly once. Each model is at its own chosen threshold.

| Model | Cut | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.4559 | 0.876 | 0.900 | 0.804 | **0.850** |
| Complement Naive Bayes | 0.5222 | 0.876 | 0.900 | 0.804 | 0.850 |
| Logistic Regression | 0.4681 | 0.820 | 0.901 | 0.659 | 0.761 |
| Logistic Regression (balanced) + embeddings | 0.6656 | 0.879 | 0.901 | 0.810 | 0.853 |

Confusion matrix for the chosen model:

```
          predicted
          ham  spam
ham  [[  217    16  ]
spam  [   35   144  ]]
```

### Precision is the constraint, not the result

Every row sits at almost exactly 0.900 because that is the floor the
threshold was chosen against — not because four models independently arrived
at the same precision. Reading it as an achievement is a category error.
**Recall is the number that varies**, and it is what the models are being
compared on: how much spam each can catch without crossing the line.

Raising the floor is expensive, and the numbers are worth knowing before
anyone tries. At 0.95 the best F1 available falls to 0.704; at 0.98, to 0.475.

### The two Naive Bayes variants are tied

Not merely equal scores — an **identical confusion matrix**, the same call on
all 412 messages. The trainer reports Multinomial as best only because it
sorts first on a tie, which is ordering presented as a result. Logistic
regression on the same features is nine points of F1 behind, though, so the
corpus has stopped being separable by *anything*, which it was when it was 81
messages of obvious keywords.

### Read those numbers with the corpus in mind

The messages are **written rather than collected**. The second shard is
deliberately adversarial — obfuscation (`Fr33`, unicode lookalikes) and
near-misses like a real delivery notice or a genuine one-time passcode — and
26% of the ham carries spam vocabulary against 46% of the spam, which is
where the 16 false positives come from. But authored examples are the ones
somebody already thought of.

For numbers that generalise, swap in a collected corpus such as the UCI SMS
Spam Collection (~5,570 texts), keeping the `label,text` format.

## The optional embedding backend

`embeddings.py` stacks sentence-transformer features
(`all-MiniLM-L6-v2`, 384 dimensions) beside the 4,479 tf-idf columns. It is
**off by default**, and the reason is the measurement rather than taste:

```bash
python tools/compare_backends.py --sweep-c
```

| features | estimator | precision | recall | F1 |
|---|---|---|---|---|
| tf-idf | MultinomialNB *(default)* | 0.900 | 0.804 | **0.850** |
| embeddings | LogReg | 0.903 | 0.726 | 0.805 |
| embeddings | RBF SVM | 0.913 | 0.765 | 0.833 |
| tf-idf + embeddings | LogReg balanced | 0.901 | 0.810 | **0.853** |

The transformer is worth **+0.003 F1**, and the confusion matrices say what
that means in plain terms: *one more spam caught out of 179*, for about two
gigabytes of dependencies and an 88 MB model. On its own it is **worse** than
tf-idf — 0.833 against 0.850 — which is the expected result and worth stating
rather than burying. MiniLM was trained on general web text and has never
seen a smishing corpus; tf-idf learns this corpus's own vocabulary directly,
and `moneytoken`, `shortcodetoken` and "claim" are strong enough evidence
that a general-purpose sentence encoder dilutes it.

It is here anyway, because the gap should widen as the corpus grows —
embeddings generalise to paraphrases a lexical model has never seen — and
because that is a claim with an expiry date, so the code to re-check it needs
to exist.

```bash
pip install -r requirements-embeddings.txt
# fetch the weights: the curl commands are in that file
python train_spam_classifier.py --embeddings
```

Two properties it keeps:

- **The model is local.** `embeddings.py` loads from a directory, never a hub
  id. A filter that reads someone's text messages should not need the network
  to do it, and nothing in it sends a message anywhere.
- **Nothing imports torch unless you ask.** The import is inside
  `load_encoder`, so `import spamlib` stays as cheap as it was and the
  default path runs with none of the extras installed. `available()` answers
  with `importlib.util.find_spec`, which does not execute the package.

With the backend loaded, the web UI's word-by-word explanation stops at the
lexical columns. The other 384 carry no per-word meaning — "dimension 137
contributed +0.04" is a number with a label on it, not an explanation.

## Extending it

- **More data** — add a CSV to `data/`, keeping the `label,text` format.
  Shards are read in filename order and duplicate texts are dropped, so a
  message repeated across two files cannot be both trained on and tested
  against.
- **Other models** — add to `candidate_models()` in `spamlib.py`; the
  comparison and the threshold sweep pick it up. It needs `predict_proba`,
  which is what rules out `LinearSVC` however well it scores.
- **Serve it** — `app.py` is the reference for wrapping the model in HTTP.

## Note on the saved model

`spam_model.joblib` and `vectorizer.joblib` are git-ignored. Pickled
scikit-learn objects are tied to the version that wrote them and frequently
refuse to load under a different release, so committing them would ship a
file that breaks elsewhere. Training takes under a second — just rerun.

## License

[MIT](LICENSE) — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and test
conventions.
