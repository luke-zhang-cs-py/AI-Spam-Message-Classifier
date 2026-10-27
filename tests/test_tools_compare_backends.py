"""tools/compare_backends.py, the measurement behind `--embeddings` being off.

The default install has no torch and no encoder weights, and CI has neither,
so the real comparison cannot run here. What can run is everything around
the encoder: the fold features, the table, the choice of the best row, the
difference it prints against the default model, and the C sweep. The
encoder is replaced the way `test_degraded_paths.py` replaces it -- a
stand-in with no torch behind it -- and the corpus is cut down to a
stratified sixty messages so the RBF row does not dominate the suite.

The stand-in is not constant. A fixed vector for every message would make
the embedding rows a coin toss and leave nothing for the table to rank; this
one counts a few spam words, so the dense half carries real signal and the
classifiers have something to separate.
"""
import os
import re
import sys
import types
import warnings

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import clone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import compare_backends as compare                        # noqa: E402
from pipeline import embeddings                           # noqa: E402
from pipeline import spamlib                              # noqa: E402

KEYWORDS = ("free", "win", "prize", "claim", "call", "text", "cash", "now")


class KeywordEncoder:
    """Embeds a message as counts of a handful of spam words, plus a bias."""

    def get_sentence_embedding_dimension(self):
        return len(KEYWORDS) + 1

    def encode(self, texts, **_ignored):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([lowered.count(word) for word in KEYWORDS] + [1.0])
        vectors = np.asarray(rows, dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def small_corpus(monkeypatch):
    """Thirty of each label from the real corpus, and the stand-in encoder."""
    full = spamlib.load_data()
    frame = full.groupby("label", group_keys=False).head(30)
    frame = frame.reset_index(drop=True)
    monkeypatch.setattr(spamlib, "load_data", lambda path=None: frame)
    monkeypatch.setattr(spamlib, "embedding_backend_problem",
                        lambda model_dir=None: None)
    monkeypatch.setattr(embeddings, "load_encoder",
                        lambda model_dir=None: KeywordEncoder())
    # Two readings of the clock, sixty-three seconds apart. The tool gets a
    # clock of its own: patching time.time itself would hand these two ticks
    # to joblib inside scikit-learn as well, which reads the clock too.
    ticks = iter([1000.0, 1063.0])
    monkeypatch.setattr(compare, "time",
                        types.SimpleNamespace(time=lambda: next(ticks)))
    return frame


def table_rows(printed):
    """(features, estimator, threshold, precision, recall, F1) per row."""
    rows = []
    for line in printed.splitlines():
        found = re.match(r"  (tf-idf \+ embeddings|tf-idf|embeddings)\s+"
                         r"(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$",
                         line)
        if found:
            rows.append((found.group(1), found.group(2).strip())
                        + tuple(float(v) for v in found.groups()[2:]))
    return rows


def test_without_the_extras_it_says_why_and_exits_one(monkeypatch, capsys):
    """The state of CI and of every default install. It used to be the only
    state anyone had run this in, by accident; now it is asserted, and the
    corpus is never loaded on the way to the refusal."""
    monkeypatch.setattr(spamlib, "embedding_backend_problem",
                        lambda model_dir=None: "sentence-transformers is "
                                               "not installed")
    loaded = []
    monkeypatch.setattr(spamlib, "load_data",
                        lambda path=None: loaded.append(1))

    assert compare.main([]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == ("cannot compare: sentence-transformers is "
                                    "not installed")
    assert captured.out == ""
    assert loaded == [], "it loaded the corpus before checking it could run"


def test_the_fold_features_are_refitted_with_either_half_or_both(
        small_corpus):
    """One fold per CV split, and the width each backend should give: the
    lexical vocabulary, the encoder's dimension, or the two side by side."""
    y = small_corpus["label"].map(spamlib.LABELS).to_numpy()
    lexical = compare.features(small_corpus, y, lexical=True, semantic=False)
    dense = compare.features(small_corpus, y, lexical=False, semantic=True)
    both = compare.features(small_corpus, y, lexical=True, semantic=True)

    width = KeywordEncoder().get_sentence_embedding_dimension()
    assert len(lexical) == len(dense) == len(both) == spamlib.CV_FOLDS
    for (_, _, words, _), (_, _, vectors, _), (_, _, stacked, _) in zip(
            lexical, dense, both):
        assert issparse(words) and issparse(stacked)
        assert vectors.shape[1] == width
        assert stacked.shape[1] == words.shape[1] + width, (
            "the combined features are not the two halves side by side")


def test_the_rbf_row_is_platt_scaled_without_the_deprecated_flag(
        small_corpus):
    """The row was `SVC(probability=True)`, deprecated in scikit-learn 1.9
    and gone in 1.11. What replaces it has to keep what the row relied on:
    one RBF SVM at EMBEDDING_C, and probabilities that rank the held-out
    messages exactly as that SVM's decision function does, since the
    threshold is chosen on that ranking."""
    y = small_corpus["label"].map(spamlib.LABELS).to_numpy()
    train, _, X_train, X_test = compare.features(
        small_corpus, y, lexical=False, semantic=True)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        fitted = clone(compare.rbf_svm(spamlib.EMBEDDING_C)).fit(
            X_train, y[train])

    assert len(fitted.calibrated_classifiers_) == 1, (
        "one SVM on every training row, as libsvm fitted, not an ensemble")
    svm = fitted.calibrated_classifiers_[0].estimator
    assert svm.kernel == "rbf" and svm.C == spamlib.EMBEDDING_C
    assert svm.probability is not True, "the deprecated flag is back"
    spam = fitted.predict_proba(X_test)[:, 1]
    assert ((spam >= 0.0) & (spam <= 1.0)).all()
    ranked = spam[np.argsort(svm.decision_function(X_test), kind="stable")]
    assert (np.diff(ranked) >= 0).all(), (
        "the probabilities reorder what the SVM's decision function ranks")


# The deprecated flag warns on every fit, so this fails while any row of the
# table still builds an SVC with it.
@pytest.mark.filterwarnings("error:The `probability` parameter:FutureWarning")
def test_the_comparison_prints_every_row_and_picks_the_best(small_corpus,
                                                            capsys):
    assert compare.main([]) == 0
    printed = capsys.readouterr().out

    assert printed.startswith("60 messages, ")
    assert "floor: precision >= %.2f, %d-fold, seed %d" % (
        spamlib.PRECISION_FLOOR, spamlib.CV_FOLDS,
        spamlib.RANDOM_STATE) in printed

    rows = table_rows(printed)
    expected = ([("tf-idf", name) for name in
                 ("MultinomialNB", "ComplementNB", "LogReg")]
                + [("embeddings", "LogReg"), ("embeddings", "RBF SVM")]
                + [("tf-idf + embeddings", name)
                   for name in spamlib.candidate_models(embeddings=True)])
    assert [row[:2] for row in rows] == expected, "a row is missing or extra"
    for row in rows:
        assert 0.0 <= row[2] <= 1.0 and 0.0 <= row[5] <= 1.0, row

    # The summary, checked against the table it summarises. The table is
    # rounded to three places and the choice is made on the unrounded
    # figures, so the best row is any row printed with the top F1.
    top = max(row[5] for row in rows)
    best = re.search(r"best      : (.{19}) (.{30}) F1 ([\d.]+)", printed)
    assert best, "no best line"
    assert float(best.group(3)) == top
    assert (best.group(1).strip(), best.group(2).strip()) in [
        row[:2] for row in rows if row[5] == top]
    default = [r for r in rows if r[:2] == ("tf-idf", "MultinomialNB")][0]
    assert re.search(r"default   : tf-idf\s+MultinomialNB\s+F1 %.3f"
                     % default[5], printed)
    stated = float(re.search(r"difference: ([+-][\d.]+) F1", printed).group(1))
    assert stated == pytest.approx(top - default[5], abs=1e-3)
    assert stated >= 0, "the best row cannot score below the default"

    assert "one spam changing side moves F1" in printed
    assert "EMBEDDING_C" not in printed, "the sweep ran without --sweep-c"
    assert printed.rstrip().endswith("measured in 63s")


def test_sweep_c_scores_each_candidate_and_marks_the_constant_in_use(
        small_corpus, capsys):
    assert compare.main(["--sweep-c"]) == 0
    printed = capsys.readouterr().out
    sweep = printed.split("EMBEDDING_C, balanced logistic regression", 1)
    assert len(sweep) == 2, "the sweep did not run"

    swept = re.findall(r"^\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                       r"(  <- EMBEDDING_C)?$", sweep[1], re.M)
    assert [float(row[0]) for row in swept] == [0.1, 1.0, 3.0, 10.0, 30.0,
                                                100.0]
    marked = [float(row[0]) for row in swept if row[4]]
    assert marked == [spamlib.EMBEDDING_C], (
        "the marker should sit on the constant spamlib actually uses")
