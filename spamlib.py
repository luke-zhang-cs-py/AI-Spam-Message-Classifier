"""
spamlib.py
-----------
The one implementation of the spam pipeline.

There used to be two. `spam_classifier_all_in_one.py` carried `clean_text`,
`load_data`, `train_and_evaluate`, `save_model`, `load_artifacts` and
`predict_message`; `train_spam_classifier.py` carried its own copies of the
same six; `classify.py` imported from the second and `app.py` from the first.
A test asserted the copies agreed, which kept them in step without ever
reducing the number of copies -- so every change to the pipeline was three
edits and a test that checked you had made them all.

They now all import from here.

What changed with the merge, and why
===================================

**The dataset is a directory.** `data/*.csv`, read and concatenated in
filename order, so new material is a new file rather than an edit to a
growing one. `dataset.csv` is still read if the directory is absent, so an
older checkout keeps working.

**Digits and links are kept, as tokens for their kind.** The old
`clean_text` deleted every URL and every digit before the vectoriser saw the
message, which is ordinary advice for topic classification and close to
backwards here: in "text WIN to 80086" the short code *is* the evidence.
Deleting it leaves "text WIN to". But the literal digits are useless too --
a different number in every message appears once and generalises to
nothing -- so each is replaced by a placeholder for its kind. Measured on
the 412-message corpus, that moved F1 from 0.768 to 0.784.

**The decision threshold is chosen, not assumed.** Predicting at 0.5 leaves
the most consequential dial untouched. Blocking a real message costs the
reader something they wanted; missing a scam costs them an annoyance, so
precision is the priority -- but "maximise precision" alone is a filter that
flags nothing. So the threshold is the one that maximises F1 subject to
precision staying at or above PRECISION_FLOOR, chosen on out-of-fold
predictions so no message helps choose the cut that judges it.

On the same corpus: MultinomialNB at the default 0.5 scores F1 0.785 at
precision 0.924; at the chosen cut it scores F1 0.842 at precision 0.904.
Ten points of recall for two of precision. Raising the floor is expensive
and the numbers are worth knowing before anyone tries: at 0.95 the best F1
available falls to 0.704, and at 0.98 to 0.475.

**The threshold travels with the model.** `train_and_evaluate` returns the
estimator wrapped in `Thresholded`, whose `predict` applies the cut. Callers
keep calling `model.predict(vectorizer.transform(...))` and get the tuned
answer; nothing had to learn about thresholds, and an artifact loaded from
disk cannot forget the one it was chosen with.

**There is a second, optional feature backend.** `embeddings=True` stacks
sentence-transformer columns beside the tf-idf ones. It is off by default
because it was measured and is worth about +0.003 F1 -- see the numbers in
`embeddings.py`. The two backends are interchangeable through
`wants_raw_text`/`vectorize`/`frame_features` below, so nothing outside this
module needs to know which one is loaded; in particular `classify.py` reads
an embedding-backed artifact without having heard of embeddings.
"""

import glob
import os
import re
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB, MultinomialNB

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(_HERE, "data")
LEGACY_DATA_PATH = os.path.join(_HERE, "dataset.csv")
MODEL_PATH = os.path.join(_HERE, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(_HERE, "vectorizer.joblib")

# Vectoriser. Bigrams because "click here" and "free entry" carry more than
# either word alone; min_df=1 because the corpus is small enough that a term
# appearing once is still worth a feature; sublinear_tf because a message
# repeating "FREE FREE FREE" should not score three times as spammy.
NGRAM_RANGE = (1, 2)
MIN_DF = 1
STOP_WORDS = None          # "free", "win" and "call" are the signal here

# Model selection. TEST_SIZE is not used by the trainer any more -- model
# choice is cross-validated -- but app.py still takes a hold-out split for
# the figures it shows, and both façade modules re-export this so the two
# cannot drift to different numbers.
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5
MAX_ITER = 2000

# Inverse regularisation for the estimators that sit on embedding features.
# Not a taste, and not a round number picked because it looks like one:
# embeddings are unit-norm, so each of the 384 values is small, and the
# default C=1.0 penalises the weights they need into uselessness. Swept by
# `python tools/compare_backends.py --sweep-c`, F1 at the precision floor:
#
#     C     0.1    1.0    3.0   10.0   30.0  100.0
#     F1  0.655  0.760  0.835  0.853  0.856  0.853
#
# So the default would cost nine points of F1, and everything from 10 up is
# one plateau. 30 is a thousandth better and that thousandth is noise on 412
# messages; 10 is the shoulder, which is the defensible place to stand.
EMBEDDING_C = 10.0

# The precision the filter is held to. See the note above on what raising it
# costs; this is the knee of the curve on the current corpus.
PRECISION_FLOOR = 0.90

LABELS = {"ham": 0, "spam": 1}


# ---------------------------------------------------------------------------
# Normalising
# ---------------------------------------------------------------------------
_URL = re.compile(r"(?:https?://|www\.)\S+|\b\S+\.(?:com|net|org|co|io|info|ly)\b/?\S*",
                  re.I)
_MONEY = re.compile(r"[£$€]\s?\d[\d,.]*")
_SHORTCODE = re.compile(r"\b\d{4,6}\b")
_PHONE = re.compile(r"\b0\d{8,10}\b")
_DIGITS = re.compile(r"\b\d[\d,.]*\b")
_SPACE = re.compile(r"\s+")

_PUNCTUATION = str.maketrans("", "", string.punctuation)

# How shouty a message has to be before the capitals are treated as a
# feature rather than as someone typing quickly.
SHOUT_RATIO = 0.3


def clean_text(text: str) -> str:
    """Normalise a message for the vectoriser.

    Keeps the *shape* of the noisy parts and throws away their specifics: a
    link becomes `urltoken`, an amount `moneytoken`, a five-digit short code
    `shortcodetoken`. The old version of this function deleted all of them,
    which removed the most informative thing in a smishing text.

    Capitals are folded, but a message that was mostly capitals says so with
    a token, because SHOUTING is a real signal and lowercasing it silently
    discards one.
    """
    text = "" if text is None else str(text)

    letters = [c for c in text if c.isalpha()]
    shouting = bool(letters) and (
        sum(1 for c in letters if c.isupper()) / len(letters) > SHOUT_RATIO)

    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _SHORTCODE.sub(" shortcodetoken ", text)
    text = _DIGITS.sub(" numtoken ", text)
    text = text.translate(_PUNCTUATION)
    text = _SPACE.sub(" ", text).strip()

    return (text + " allcapstoken") if shouting else text


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def data_files(directory=None):
    """Every corpus shard, in filename order.

    Sorted rather than glob order so a run is reproducible: the split is
    seeded, and a seed only helps if the rows arrive in the same order.
    """
    return sorted(glob.glob(os.path.join(directory or DATA_DIR, "*.csv")))


def load_data(path=None) -> pd.DataFrame:
    """The corpus, normalised, with a `clean_text` column.

    `path` may be a directory of shards or a single CSV. With nothing given
    it prefers the shard directory and falls back to the original
    dataset.csv, so a checkout without `data/` still trains.
    """
    if path and os.path.isfile(path):
        frames = [pd.read_csv(path)]
    else:
        shards = data_files(path)
        if shards:
            frames = [pd.read_csv(shard) for shard in shards]
        elif os.path.isfile(LEGACY_DATA_PATH):
            frames = [pd.read_csv(LEGACY_DATA_PATH)]
        else:
            raise FileNotFoundError(
                "no corpus: expected CSV shards in %s or a file at %s"
                % (DATA_DIR, LEGACY_DATA_PATH))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(LABELS)]
    df["clean_text"] = df["text"].map(clean_text)
    # A message repeated across shards would otherwise be both trained on
    # and tested against, which flatters every number downstream.
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The threshold, and a model that remembers it
# ---------------------------------------------------------------------------
class Thresholded:
    """An estimator that predicts at a chosen cut rather than at 0.5.

    Wrapping rather than subclassing, and deliberately minimal: `predict`,
    `predict_proba` and `classes_` are the whole surface the rest of this
    project touches. It is picklable, so the cut is saved and loaded with
    the model instead of living in a constant somebody has to remember to
    apply -- which is how a tuned threshold quietly stops being applied.
    """

    def __init__(self, estimator, threshold=0.5, name=None):
        self.estimator = estimator
        self.threshold = float(threshold)
        self.name = name or type(estimator).__name__
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def __getattr__(self, name):
        """Anything else, ask the estimator.

        The web app explains a verdict by reading `feature_log_prob_` or
        `coef_` off the model. Wrapping it hid those, so the explanation
        came back empty -- a silent loss of the one thing the web version
        has over the CLI. Delegating keeps the wrapper transparent to code
        that legitimately wants the fitted estimator's internals.

        Guarded against the recursion that bites every __getattr__ on a
        picklable object: unpickling sets __dict__ directly, so `estimator`
        can be absent while this is being called.
        """
        if name in ("estimator", "__setstate__", "__getstate__"):
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def __repr__(self):
        return "Thresholded(%s, threshold=%.4f)" % (self.name, self.threshold)


def choose_threshold(scores, y, floor=PRECISION_FLOOR):
    """The cut with the best F1 whose precision is at least `floor`.

    Returns (threshold, precision, recall, f1). If the floor cannot be
    reached at any cut -- a corpus the model simply cannot separate that
    cleanly -- it falls back to the best F1 outright rather than returning a
    filter that flags nothing, and the caller can see the precision it got.
    """
    y = np.asarray(y)
    best = None
    fallback = None

    for cut in np.unique(scores):
        flagged = (scores >= cut).astype(int)
        if flagged.sum() == 0:
            continue
        precision = precision_score(y, flagged, zero_division=0)
        recall = recall_score(y, flagged, zero_division=0)
        f1 = f1_score(y, flagged, zero_division=0)

        if fallback is None or f1 > fallback[3]:
            fallback = (float(cut), precision, recall, f1)
        if precision >= floor and (best is None or f1 > best[3]):
            best = (float(cut), precision, recall, f1)

    return best or fallback or (0.5, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_vectorizer(embeddings=False, model_dir=None):
    """The feature extractor: tf-idf, or tf-idf plus sentence embeddings.

    Imported inside the branch, not at module scope, so that the two
    gigabytes of torch behind the optional backend are only touched by
    someone who asked for them. `import spamlib` stays as cheap as it was.
    """
    tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF,
                            stop_words=STOP_WORDS, sublinear_tf=True)
    if not embeddings:
        return tfidf

    import embeddings as backend
    return backend.SemanticFeatures(lexical=tfidf, cleaner=clean_text,
                                    model_dir=model_dir)


# ---------------------------------------------------------------------------
# Feeding either backend
# ---------------------------------------------------------------------------
# The two extractors want the text in different forms -- tf-idf wants it
# normalised, the encoder wants it as written -- and the three helpers below
# are the whole of what the rest of the project needs to know about that.
# Everything routes through them, so adding the second backend did not mean
# an `if` at every call site.

def wants_raw_text(vectorizer):
    """Whether this extractor wants the message as written.

    Asked by attribute rather than by isinstance so that no caller has to
    import the embeddings module to find out -- which would defeat keeping
    torch out of the default path.
    """
    return bool(getattr(vectorizer, "wants_raw_text", False))


def frame_features(vectorizer, df, fit=False):
    """Features for a whole corpus frame, from the column this backend wants."""
    column = "text" if wants_raw_text(vectorizer) else "clean_text"
    texts = df[column].tolist()
    return vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)


def vectorize(vectorizer, messages):
    """Features for raw messages, normalising them first if that is wanted.

    Callers used to spell this `vectorizer.transform([clean_text(m)])`, which
    hard-codes the assumption that the extractor wants cleaned text. With the
    embedding backend loaded that quietly hands the encoder pre-mangled input
    and costs most of what it was added for, with no error to show for it.
    """
    messages = ["" if m is None else str(m) for m in messages]
    if wants_raw_text(vectorizer):
        return vectorizer.transform(messages)
    return vectorizer.transform([clean_text(m) for m in messages])


def embedding_backend_problem(model_dir=None):
    """Why `embeddings=True` cannot be honoured, as a sentence, or None.

    Both command-line entry points ask this before training so that
    `--embeddings` without the extras installed prints the pip command
    instead of a traceback. It lives here rather than in each CLI for the
    reason the rest of this module exists: two copies of an error message
    is how they come to say different things.

    The ImportError branch is not decoration. This project's history is of
    files being copied out to stand alone -- `spam_classifier_all_in_one.py`
    was named for it -- and a `spamlib.py` lifted without its sibling should
    say so rather than fail on a bare import line.
    """
    try:
        import embeddings as backend
    except ImportError as problem:
        return "the embeddings module is not importable: %s" % problem
    return backend.missing_requirement(model_dir)


def candidate_models(embeddings=False):
    """The estimators worth comparing, and why these.

    All of them give a calibrated-enough probability for a threshold sweep to
    mean something, which rules out LinearSVC however well it scores.
    ComplementNB earns its place by being built for imbalanced text;
    logistic regression is here as the linear baseline that is not a
    naive-Bayes variant. Tree ensembles were measured and are poor on this
    shape of data -- sparse, high-dimensional, few rows -- at recall 0.458,
    so they are not in the list.

    With embeddings the list has to change, and not by preference: both
    naive-Bayes variants model a feature as a count and reject a negative
    one outright, while half of every unit-norm embedding is negative. They
    would not score badly, they would raise. So the embedding list is the
    estimators that accept signed dense features.

    An RBF kernel was the obvious fourth candidate -- the one thing a dense
    semantic space might offer over a sparse lexical one is structure a
    linear boundary cannot use -- and it was measured rather than assumed.
    On embeddings alone it is the best of the three (F1 0.833 against 0.805
    for logistic regression), and on the combined features this backend
    actually builds it is the worst (0.845 against 0.853). It was also most
    of the 90 seconds the comparison took, because `probability=True` fits a
    Platt calibration by internal cross-validation inside every outer fold,
    and that parameter is deprecated in scikit-learn 1.9 besides. Measured,
    beaten, and removed.
    """
    if embeddings:
        return {
            "Logistic Regression (balanced)": LogisticRegression(
                max_iter=MAX_ITER, C=EMBEDDING_C, class_weight="balanced"),
            "Logistic Regression": LogisticRegression(
                max_iter=MAX_ITER, C=EMBEDDING_C),
        }
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=MAX_ITER),
    }


def _out_of_fold_scores(estimator, X, y):
    """P(spam) for every row, from a fold that did not train on it."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    return cross_val_predict(estimator, X, y, cv=cv,
                             method="predict_proba")[:, 1]


def compare_models(df, verbose=True, embeddings=False):
    """Every candidate, scored out of fold at its own chosen threshold.

    Cross-validated rather than a single hold-out split: on a corpus this
    size one split moves the F1 by several points depending on which
    messages land in it, and picking a model on that is picking a split.
    """
    vectorizer = build_vectorizer(embeddings=embeddings)
    X = frame_features(vectorizer, df, fit=True)
    y = df["label"].map(LABELS).to_numpy()

    results = []
    for name, estimator in candidate_models(embeddings=embeddings).items():
        scores = _out_of_fold_scores(estimator, X, y)
        threshold, precision, recall, f1 = choose_threshold(scores, y)
        results.append({"name": name, "estimator": estimator,
                        "threshold": threshold, "precision": precision,
                        "recall": recall, "f1": f1})

    results.sort(key=lambda r: (-r["f1"], -r["precision"]))

    if verbose:
        print("=" * 72)
        print("MODEL COMPARISON  (%d messages, %d-fold, precision floor %.2f)"
              % (len(df), CV_FOLDS, PRECISION_FLOOR))
        print("features: %s" % describe_features(vectorizer))
        print("=" * 72)
        print("  %-30s %9s %10s %8s %8s"
              % ("model", "threshold", "precision", "recall", "F1"))
        print("  " + "-" * 70)
        for row in results:
            print("  %-30s %9.4f %10.3f %8.3f %8.3f"
                  % (row["name"], row["threshold"], row["precision"],
                     row["recall"], row["f1"]))
        print()

    return results, vectorizer, X, y


def describe_features(vectorizer):
    """One line naming the backend and its width, for the training report.

    Worth printing: the two backends produce different tables and a reader
    comparing two runs needs to know which one they are looking at. Asked
    by attribute so this works for a plain TfidfVectorizer too.
    """
    lexical = getattr(vectorizer, "lexical_width", None)
    dimension = getattr(vectorizer, "dimension", None)
    if lexical is None or dimension is None:
        return "tf-idf %s-grams on normalised text" % (NGRAM_RANGE,)
    return ("tf-idf (%d terms) + %d embedding dimensions from %s"
            % (lexical, dimension,
               os.path.basename(getattr(vectorizer, "model_dir", "?"))))


def train_and_evaluate(df, verbose=True, embeddings=False):
    """Pick a model and a threshold, fit on everything, return both.

    Returns (model, vectorizer) to match what every caller already expects.
    The model is a `Thresholded`, so `predict` honours the chosen cut.
    """
    results, vectorizer, X, y = compare_models(df, verbose=verbose,
                                               embeddings=embeddings)
    best = results[0]

    estimator = best["estimator"]
    estimator.fit(X, y)
    model = Thresholded(estimator, best["threshold"], name=best["name"])

    if verbose:
        print("Chosen: %s at threshold %.4f" % (best["name"], best["threshold"]))
        print("  out-of-fold precision %.3f, recall %.3f, F1 %.3f"
              % (best["precision"], best["recall"], best["f1"]))
        if best["precision"] < PRECISION_FLOOR:
            print("  NOTE: no threshold reached the %.2f precision floor; "
                  "this is the best F1 available." % PRECISION_FLOOR)
        print()
        flagged = model.predict(X)
        print(classification_report(y, flagged, target_names=["ham", "spam"],
                                    zero_division=0))
        print("Confusion matrix [[TN FP] [FN TP]] (in-sample):")
        print(confusion_matrix(y, flagged))
        print()

    return model, vectorizer


def metrics(df, model, vectorizer):
    """Out-of-fold metrics for a trained pair, for the web app to display.

    Out of fold rather than in sample: an in-sample score on a corpus this
    small reads about ten points high, and a figure on a page that flatters
    the model is worse than no figure.
    """
    X = frame_features(vectorizer, df)
    y = df["label"].map(LABELS).to_numpy()

    estimator = getattr(model, "estimator", model)
    threshold = getattr(model, "threshold", 0.5)
    scores = _out_of_fold_scores(estimator, X, y)
    flagged = (scores >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y, flagged)), 4),
        "precision": round(float(precision_score(y, flagged, zero_division=0)), 4),
        "recall": round(float(recall_score(y, flagged, zero_division=0)), 4),
        "f1": round(float(f1_score(y, flagged, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
        "confusionMatrix": confusion_matrix(y, flagged).tolist(),
        "evaluatedOn": "out-of-fold predictions, %d folds" % CV_FOLDS,
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def save_model(model, vectorizer, model_path=None, vectorizer_path=None):
    import joblib
    joblib.dump(model, model_path or MODEL_PATH)
    joblib.dump(vectorizer, vectorizer_path or VECTORIZER_PATH)


def load_artifacts(model_path=None, vectorizer_path=None):
    """(model, vectorizer), or (None, None) if either is missing.

    Never exits. An earlier version called sys.exit(1) from in here, which
    made it untestable and meant importing it handed the caller a function
    able to end their process. What to do about a missing model is main()'s
    decision.
    """
    import joblib
    try:
        return (joblib.load(model_path or MODEL_PATH),
                joblib.load(vectorizer_path or VECTORIZER_PATH))
    except (OSError, EOFError, ValueError, AttributeError, ModuleNotFoundError):
        return None, None


def predict_message(message: str, model, vectorizer) -> str:
    """"spam" or "ham" for one message."""
    vector = vectorize(vectorizer, [message])
    return "spam" if int(model.predict(vector)[0]) == 1 else "ham"


def spam_probability(message: str, model, vectorizer) -> float:
    """P(spam) for one message, before the threshold is applied.

    Worth exposing separately: "0.63, and the cut is 0.47" tells a reader
    something that "spam" does not.
    """
    vector = vectorize(vectorizer, [message])
    return float(model.predict_proba(vector)[0][1])
