"""The paths that only a degraded input reaches.

Every test here was written by starting from an uncovered line and asking
the question first: *is this reachable at all?* Three answers came back, and
they are treated differently.

**Reachable and ordinary.** The legacy `dataset.csv` fallback is what a
checkout with no `data/` directory takes, and the missing-corpus error is
what one with neither takes. Both are real states of a real clone, and
neither had a test.

**Reachable only by a caller passing something broken.** The
`flagged.sum() == 0` guard in `choose_threshold` cannot fire for any cut
drawn from `np.unique(scores)` -- the largest cut always flags at least the
row it came from -- unless the scores contain NaN, because every comparison
against NaN is False. A model that emits NaN probabilities is broken, but
`choose_threshold` is exported and a caller can pass anything, so the guard
is what keeps it total rather than crashing on the way to a confusing
answer. Tested with the input that reaches it, and documented as the only
one that does.

**Not reachable, and therefore not tested.** Nothing here covers a process
entry point; `if __name__ == "__main__":` blocks are excluded in
`.coveragerc`, because a module that started a Flask server on import could
not be imported by a test at all. What those blocks *call* is covered
directly in `tests/test_cli_entry_points.py`.
"""
import contextlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import embeddings                          # noqa: E402
import spamlib                             # noqa: E402

# Two words per message and none of them one character long: the default
# `token_pattern` wants at least two, so a fixture of ["a b", "c d"] gives
# TfidfVectorizer an empty vocabulary and a ValueError that has nothing to do
# with the test using it.
WORDS = ["free prize now", "see you at six"]


class StandInEncoder:
    """A fixed-width encoder that needs no torch.

    Named for what it is rather than what it holds -- an earlier version of
    this file called it `Fixed`, which says nothing about its job.
    """

    def __init__(self, width):
        self.width = width

    def get_sentence_embedding_dimension(self):
        return self.width

    def encode(self, texts, **_ignored):
        row = np.linspace(-1, 1, self.width)
        return np.tile(row, (len(texts), 1)).astype("float32")


@contextlib.contextmanager
def _fake_encoder(width):
    """Swap in the stand-in, and put the real loader back afterwards.

    A context manager rather than the same four lines of save-and-restore in
    every test: the cache has to be cleared on the way out as well, and an
    exception between the swap and the restore would otherwise leave every
    later test holding the stand-in.
    """
    original = embeddings.load_encoder
    embeddings.load_encoder = lambda model_dir=None: StandInEncoder(width)
    try:
        yield
    finally:
        embeddings.load_encoder = original
        embeddings.reset()


# ------------------------------------------------------- finding the corpus

def test_a_checkout_with_no_shard_directory_falls_back_to_dataset_csv(
        tmp_path, monkeypatch):
    """The original single-file corpus is still read when `data/` is absent,
    so an older checkout keeps training. That is the whole promise of
    keeping dataset.csv around, and nothing checked it."""
    legacy = tmp_path / "dataset.csv"
    legacy.write_text("label,text\nspam,win a free prize\nham,see you at six\n",
                      encoding="utf-8")
    monkeypatch.setattr(spamlib, "DATA_DIR", str(tmp_path / "no-shards-here"))
    monkeypatch.setattr(spamlib, "LEGACY_DATA_PATH", str(legacy))

    df = spamlib.load_data()
    assert len(df) == 2
    assert set(df["label"]) == {"spam", "ham"}
    assert "clean_text" in df.columns


def test_a_checkout_with_no_corpus_at_all_says_where_it_looked(tmp_path,
                                                               monkeypatch):
    """Both absent. The message names both places, because "no corpus" on
    its own does not tell anyone what to do about it."""
    monkeypatch.setattr(spamlib, "DATA_DIR", str(tmp_path / "nothing"))
    monkeypatch.setattr(spamlib, "LEGACY_DATA_PATH", str(tmp_path / "none.csv"))

    with pytest.raises(FileNotFoundError) as raised:
        spamlib.load_data()
    message = str(raised.value)
    assert "nothing" in message and "none.csv" in message


def test_the_real_dataset_csv_is_still_loadable():
    """The fallback is only a promise if the file it falls back to works.
    It is committed for exactly this reason -- CI has no .joblib artifacts,
    so the first test needing a model trains one."""
    df = spamlib.load_data(spamlib.LEGACY_DATA_PATH)
    assert len(df) > 0
    assert set(df["label"]) == {"spam", "ham"}


# ------------------------------------------------ the threshold, degenerate

def test_a_cut_that_flags_nothing_is_skipped_rather_than_scored():
    """The only input that reaches the `flagged.sum() == 0` guard.

    For finite scores it cannot fire: `np.unique(scores)` contains the
    maximum, and `scores >= max` is True for at least that row. NaN is the
    exception, because every comparison against it is False -- so a NaN in
    the score vector produces a candidate cut that flags nothing at all.

    Without the guard this still would not crash (`zero_division=0` handles
    the precision) but it would put a 0/0/0 row into the running best. The
    assertion is that a usable threshold comes back anyway.
    """
    scores = np.array([np.nan, 0.1, 0.4, 0.9, 0.95])
    y = np.array([1, 0, 0, 1, 1])

    cut, precision, recall, f1 = spamlib.choose_threshold(scores, y, floor=0.5)
    assert np.isfinite(cut), "a NaN cut was chosen"
    assert f1 > 0.0, "no threshold was usable"
    assert precision >= 0.5


def test_the_floor_falls_back_to_the_best_f1_when_it_cannot_be_reached():
    """A corpus that cannot be separated to the floor should return the best
    filter available, not one that flags nothing. `best or fallback` is the
    line, and a filter with no recall is the failure it avoids.

    The fixture has to be built rather than guessed at. The first attempt
    here was scores [0.4, 0.5, 0.6, 0.5, 0.4] against y [1, 0, 1, 0, 0],
    which looks unseparable and is not: the cut at 0.6 flags exactly one
    row, that row is spam, and precision is 1.0. The single highest-scoring
    message has to be *ham* for no cut to reach the floor.
    """
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([0, 1, 1, 1])

    cut, precision, recall, f1 = spamlib.choose_threshold(scores, y, floor=0.9)
    assert precision < 0.9, "this fixture was supposed to be unreachable"
    assert recall > 0.0, "it returned a filter that flags nothing"
    assert f1 > 0.0


def test_a_score_vector_with_no_usable_cut_returns_a_neutral_default():
    """The last resort, `(0.5, 0.0, 0.0, 0.0)`. Reached when the loop never
    finds a single candidate -- an empty score vector is the way in."""
    assert spamlib.choose_threshold(np.array([]), np.array([])) == (
        0.5, 0.0, 0.0, 0.0)


def test_the_trainer_says_so_when_the_floor_was_not_reached(capsys):
    """A precision figure below the floor is the one number in the report a
    reader must not skim past, so it gets a line of its own. Built from
    near-duplicate text with contradictory labels: nothing can separate it.
    """
    import pandas as pd

    rows = []
    for index in range(spamlib.CV_FOLDS * 2):
        rows.append({"label": "spam" if index % 2 else "ham",
                     "text": "hello there how are you today"})
    df = pd.DataFrame(rows)
    df["clean_text"] = df["text"].map(spamlib.clean_text)

    spamlib.train_and_evaluate(df, verbose=True)
    printed = capsys.readouterr().out
    assert "no threshold reached" in printed
    assert "%.2f" % spamlib.PRECISION_FLOOR in printed


# --------------------------------------------------------- small surfaces

def test_the_thresholded_model_says_what_it_is():
    """A repr is debugging equipment, and one that raises is worse than
    none. This one interpolates two attributes, either of which could go
    missing in a refactor of the wrapper."""
    from sklearn.naive_bayes import MultinomialNB

    wrapped = spamlib.Thresholded(MultinomialNB(), threshold=0.4559,
                                  name="Multinomial Naive Bayes")
    text = repr(wrapped)
    assert "Thresholded" in text
    assert "Multinomial Naive Bayes" in text
    assert "0.4559" in text


def test_the_features_are_sparse_whether_or_not_there_is_a_lexical_half():
    """One method, one return type.

    `_build` used to hand back the raw dense array when `lexical is None`
    and a CSR matrix otherwise, so callers lost the sparse interface
    depending on how the vectoriser had been constructed. `app.classify`
    asks the matrix for `.nnz`, which an ndarray does not have, so an
    embeddings-only vectoriser reached the web app as an AttributeError.
    Nothing wanted the dense form -- sklearn takes either -- it was just
    what `encode` happened to return.
    """
    from scipy import sparse

    with _fake_encoder(width=6):
        only = embeddings.SemanticFeatures(lexical=None).fit_transform(WORDS)
        both = embeddings.SemanticFeatures(
            lexical=spamlib.build_vectorizer(),
            cleaner=spamlib.clean_text).fit_transform(WORDS)

    assert sparse.issparse(only), "the embeddings-only matrix is not sparse"
    assert sparse.issparse(both)
    assert hasattr(only, "nnz") and hasattr(both, "nnz")


def test_an_embeddings_only_backend_has_nothing_to_explain():
    """The boundary, at its degenerate end.

    `explainable_width` tested `if width:`, and `lexical_width` is 0 when
    there is no lexical half -- so a zero was read as "plain vectoriser,
    explain every column" and the page would have listed all of the
    embedding dimensions as explainable words. The exact opposite of what
    the boundary exists for, from a falsy zero.
    """
    import app as web

    with _fake_encoder(width=6):
        features = embeddings.SemanticFeatures(lexical=None)
        matrix = features.fit_transform(WORDS)

    assert features.lexical_width == 0
    assert matrix.shape[1] == 6, "the fixture has only embedding columns"
    assert web.explainable_width(features) == 0, (
        "the embedding dimensions are being offered as explainable words")
    assert web.known_tokens(matrix[0], features) == 0, (
        "dense embedding columns are being counted as vocabulary hits")

    # And the ordinary case is unaffected: a plain tf-idf vectoriser has no
    # lexical_width attribute at all, and every column is a word.
    plain = spamlib.build_vectorizer()
    lexical = plain.fit_transform([spamlib.clean_text(w) for w in WORDS])
    assert web.explainable_width(plain) == lexical.shape[1]
    assert web.known_tokens(lexical[0], plain) == lexical[0].nnz


def test_the_embedding_features_can_skip_normalising_entirely():
    """`cleaner=None` means the lexical half gets the text as given.

    Not the configuration `spamlib.build_vectorizer` builds -- it always
    passes `clean_text` -- but the one a caller wants when the text is
    already normalised, and the branch is two lines from the one that is
    used constantly.
    """
    with _fake_encoder(width=4):
        features = embeddings.SemanticFeatures(
            lexical=spamlib.build_vectorizer(), cleaner=None)
        matrix = features.fit_transform(["text WIN to 80086",
                                         "see you at six"])

    assert matrix.shape[0] == 2

    # The shape tokens are the visible difference, and the only one worth
    # asserting: case folding is not evidence either way, because
    # TfidfVectorizer lowercases on its own whether clean_text ran or not.
    vocabulary = set(features.lexical.get_feature_names_out())
    assert "shortcodetoken" not in vocabulary, "the cleaner ran anyway"
    assert "80086" in vocabulary, "the literal short code should survive"
