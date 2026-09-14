"""The optional embedding backend.

Almost every test here runs without torch, sentence-transformers, or the 88 MB
of weights, and that is the point rather than a compromise. CI installs
`requirements.txt`, which deliberately does not include the extras -- so a
suite that could only test this backend with them installed would not test it
at all, and the optional path would rot exactly where nobody looks.

What makes that possible is `load_encoder` being one function with one job.
A fake encoder substituted for it exercises `SemanticFeatures`, the
raw-versus-cleaned routing, the pickling, the estimator list, the web app's
explanation boundary and both CLIs. The fake returns *signed* vectors,
because the interesting consequence of a real embedding -- that naive Bayes
cannot accept one -- follows from the sign and would be missed by a fake
that returned counts.

The handful of tests that need the real model say so and skip without it.
They check the two things a fake cannot: that the weights on disk load, and
that what comes out is a semantic space rather than 384 arbitrary numbers.
"""
import hashlib
import io
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import embeddings                          # noqa: E402
import spamlib                             # noqa: E402
import train_spam_classifier as trainer    # noqa: E402

FAKE_WIDTH = 8

MESSAGES = [
    "WINNER!! Claim your FREE prize now, call 09061701461",
    "Hey, are we still on for coffee at 6?",
    "Text STOP to 88888 to unsubscribe",
    "can you send me the notes from class",
]


class FakeEncoder:
    """Deterministic pseudo-embeddings, with no torch anywhere near them.

    Hash-derived and therefore meaningless, which is deliberate: a fake that
    encoded real semantics would let a test assert a score, and a score is
    the one thing that has to be measured on the real model. What this fake
    reproduces is the *shape* of the contract -- fixed width, signed values,
    unit norm when asked, one row per message, in order.
    """

    def __init__(self, width=FAKE_WIDTH):
        self.width = width
        self.seen = []

    def get_sentence_embedding_dimension(self):
        return self.width

    def encode(self, texts, batch_size=None, show_progress_bar=None,
               normalize_embeddings=False):
        self.seen.extend(texts)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.frombuffer(digest[:self.width], dtype=np.uint8)
            # Centred, so half the features are negative like a real one.
            row = row.astype("float32") - 128.0
            if normalize_embeddings:
                row = row / (np.linalg.norm(row) or 1.0)
            rows.append(row)
        return np.vstack(rows)


@pytest.fixture
def encoder(monkeypatch):
    """A fake encoder in place of the real one, and no cache left behind."""
    fake = FakeEncoder()
    monkeypatch.setattr(embeddings, "load_encoder", lambda model_dir=None: fake)
    yield fake
    embeddings.reset()


@pytest.fixture
def corpus():
    return spamlib.load_data()


# --------------------------------------------------------- staying cheap

def test_importing_the_backend_does_not_import_torch():
    """The whole reason the import is inside `load_encoder`.

    Checked in a subprocess because this process may well have imported
    torch by now for one of the tests below, and `sys.modules` here would
    say so whatever the import graph actually does.
    """
    done = subprocess.run(
        [sys.executable, "-c",
         "import embeddings, spamlib, sys;"
         "print('torch' in sys.modules, "
         "'sentence_transformers' in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-500:]
    assert done.stdout.strip() == "False False", done.stdout


def test_building_the_features_does_not_load_the_encoder(monkeypatch):
    """Construction is free; only fitting costs a second and a half."""
    def refuse(model_dir=None):
        raise AssertionError("the encoder was loaded during construction")

    monkeypatch.setattr(embeddings, "load_encoder", refuse)
    features = spamlib.build_vectorizer(embeddings=True)
    assert features.lexical is not None
    assert features.dimension is None, "nothing has been encoded yet"


# ------------------------------------------------- saying why it is off

def test_missing_requirement_names_the_install_command(monkeypatch):
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: None)
    problem = embeddings.missing_requirement()
    assert "requirements-embeddings.txt" in problem
    assert not embeddings.available()


def test_missing_requirement_names_the_directory_and_the_override(tmp_path,
                                                                  monkeypatch):
    """Installed but no weights is a different problem with a different fix,
    and "install the extras" is useless advice to someone who has."""
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: object())
    absent = str(tmp_path / "not-fetched")
    problem = embeddings.missing_requirement(absent)
    assert absent in problem
    assert embeddings.MODEL_DIR_ENV in problem
    assert "pip install" not in problem


def test_missing_requirement_notices_a_half_fetched_model(tmp_path,
                                                          monkeypatch):
    """An interrupted curl leaves a directory that exists and does not work.
    Without this check it fails deep inside the library on a missing key."""
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: object())
    partial = tmp_path / "half"
    partial.mkdir()
    (partial / "config.json").write_text("{}", encoding="utf-8")

    problem = embeddings.missing_requirement(str(partial))
    assert "not a complete" in problem
    assert "modules.json" in problem and "tokenizer.json" in problem


def test_load_encoder_says_what_to_do_instead_of_raising_importerror(
        tmp_path, monkeypatch):
    """"No module named torch" is a worse answer to "why is this off" than
    the command that turns it on.

    find_spec is pinned so this exercises the branch it names. Without the
    pin it passed here and failed in CI: on a machine with the extras
    installed the missing *directory* is the first complaint, and on one
    without them the missing *library* is -- so the assertion held only
    where the library happened to be present.
    """
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: object())
    with pytest.raises(RuntimeError) as raised:
        embeddings.load_encoder(str(tmp_path / "nothing-here"))
    assert "nothing-here" in str(raised.value)


def test_load_encoder_names_the_library_before_the_weights(monkeypatch,
                                                           tmp_path):
    """Precedence, stated. With nothing installed the answer is the pip
    command, not a path -- fetching 87 MB of weights for a library you do
    not have is the wrong first move."""
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: None)
    with pytest.raises(RuntimeError) as raised:
        embeddings.load_encoder(str(tmp_path / "nothing-here"))
    assert "requirements-embeddings.txt" in str(raised.value)
    assert "nothing-here" not in str(raised.value)


def test_the_environment_variable_redirects_the_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv(embeddings.MODEL_DIR_ENV, str(tmp_path))
    assert embeddings.model_directory() == str(tmp_path)
    # An explicit argument still wins over the environment.
    assert embeddings.model_directory("elsewhere") == "elsewhere"


def test_the_default_directory_is_absolute():
    """A relative default would resolve against whatever directory the
    caller happened to be standing in -- the bug this project already had
    with its model artifacts."""
    assert os.path.isabs(embeddings.DEFAULT_MODEL_DIR)


def test_the_backend_problem_survives_a_spamlib_lifted_on_its_own(monkeypatch):
    """`spam_classifier_all_in_one.py` is named for this project's habit of
    copying files out to stand alone. A spamlib without its sibling should
    say so rather than fail on a bare import line."""
    monkeypatch.setitem(sys.modules, "embeddings", None)
    problem = spamlib.embedding_backend_problem()
    assert "not importable" in problem


# ------------------------------------------------------- the encoder cache

def test_the_encoder_is_loaded_once_per_directory(monkeypatch, tmp_path):
    """Loading is 87 MB off disk. Twice would be twice that, and both
    `fit_transform` and every later `transform` want the same object."""
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: object())
    for name in embeddings.REQUIRED_FILES:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    loads = []

    class FakeModule:
        @staticmethod
        def SentenceTransformer(directory):
            loads.append(directory)
            return FakeEncoder()

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeModule)
    embeddings.reset()
    try:
        first = embeddings.load_encoder(str(tmp_path))
        second = embeddings.load_encoder(str(tmp_path))
        assert first is second
        assert loads == [str(tmp_path)]

        embeddings.reset()
        embeddings.load_encoder(str(tmp_path))
        assert len(loads) == 2, "reset() did not forget the encoder"
    finally:
        embeddings.reset()


def test_encoding_nothing_still_has_a_width(encoder):
    """The caller needs a width to stack against. `encode([])` is not
    reliably shaped across sentence-transformers versions."""
    empty = embeddings.encode([])
    assert empty.shape == (0, FAKE_WIDTH)


# ------------------------------------- each half gets the text it wants

def test_the_embedding_half_reads_the_message_as_written(encoder, corpus):
    """The single most losable property in this whole feature.

    `clean_text` turns "text WIN to 80086" into "text win to shortcodetoken",
    which is exactly right for a bag of words and throws away the sentence a
    transformer was trained to read. Feeding it the normalised text would
    cost most of what the backend is for and produce no error at all -- just
    a slightly worse number nobody could explain.
    """
    features = spamlib.build_vectorizer(embeddings=True)
    spamlib.frame_features(features, corpus, fit=True)

    assert encoder.seen == corpus["text"].tolist()
    # And the normalised form was genuinely different, so the assertion above
    # is not passing because the two are the same thing on this corpus.
    assert any(raw != spamlib.clean_text(raw) for raw in encoder.seen)


def test_the_lexical_half_reads_the_normalised_message(encoder, corpus):
    """The other side of the same contract: the tf-idf columns must still be
    built from cleaned text, or the shape tokens stop existing."""
    features = spamlib.build_vectorizer(embeddings=True)
    spamlib.frame_features(features, corpus, fit=True)

    vocabulary = set(features.lexical.get_feature_names_out())
    assert "shortcodetoken" in vocabulary
    assert "urltoken" in vocabulary


def test_vectorize_routes_a_single_message_by_backend(encoder):
    """`vectorize` is what replaced `transform([clean_text(m)])` at every
    call site, and this is the behaviour that justified the change."""
    plain = spamlib.build_vectorizer()
    plain.fit([spamlib.clean_text(m) for m in MESSAGES])
    assert spamlib.wants_raw_text(plain) is False
    spamlib.vectorize(plain, MESSAGES)      # cleaned on the way in

    combined = spamlib.build_vectorizer(embeddings=True)
    combined.fit_transform(MESSAGES)
    assert spamlib.wants_raw_text(combined) is True

    encoder.seen.clear()
    spamlib.vectorize(combined, ["Text WIN to 80086 now"])
    assert encoder.seen == ["Text WIN to 80086 now"]


def test_vectorize_tolerates_none_and_non_strings(encoder):
    """Reached from the web app's batch endpoint, which filters most of this
    out, and from anyone using the library directly, which does not."""
    combined = spamlib.build_vectorizer(embeddings=True)
    combined.fit_transform(MESSAGES)
    matrix = spamlib.vectorize(combined, [None, 17, "ok"])
    assert matrix.shape[0] == 3


def test_the_two_halves_are_stacked_lexical_first(encoder, corpus):
    features = spamlib.build_vectorizer(embeddings=True)
    matrix = spamlib.frame_features(features, corpus, fit=True)

    assert features.lexical_width > 0
    assert features.dimension == FAKE_WIDTH
    assert matrix.shape == (len(corpus),
                            features.lexical_width + FAKE_WIDTH)

    # The lexical block is sparse and the embedding block is dense. Stated as
    # a density comparison rather than "every embedding value is non-zero",
    # which is nearly true and not actually guaranteed -- this fake centres
    # hash bytes on 128, so a byte of exactly 128 is a real zero, and over
    # 3,296 values one turns up. Density is the property the code depends on
    # (see `app.known_tokens`); exact non-zeroness is not.
    lexical_block = matrix[:, :features.lexical_width]
    dense_block = matrix[:, features.lexical_width:]
    lexical_density = lexical_block.nnz / (lexical_block.shape[0]
                                           * lexical_block.shape[1])
    dense_density = dense_block.nnz / (dense_block.shape[0]
                                       * dense_block.shape[1])
    assert lexical_density < 0.01
    assert dense_density > 0.99


def test_transform_does_not_refit_the_vocabulary(encoder, corpus):
    """`transform` on unseen words must not grow the matrix, or the model's
    coefficients stop lining up with the columns."""
    features = spamlib.build_vectorizer(embeddings=True)
    fitted = spamlib.frame_features(features, corpus, fit=True)
    later = spamlib.vectorize(features, ["a completely unseen zxqv message"])
    assert later.shape[1] == fitted.shape[1]


def test_the_lexical_half_is_optional(encoder):
    """Embeddings alone: not reachable from spamlib's own API, but it is the
    row in `tools/compare_backends.py` that shows the backend losing to
    tf-idf on its own, so it has to keep working."""
    dense_only = embeddings.SemanticFeatures(lexical=None)
    matrix = dense_only.fit_transform(MESSAGES)
    assert matrix.shape == (len(MESSAGES), FAKE_WIDTH)
    assert list(dense_only.get_feature_names_out()) == [
        "embedding[%d]" % i for i in range(FAKE_WIDTH)]


# ---------------------------------------------- the estimator list changes

def test_naive_bayes_is_excluded_because_it_would_raise(encoder, corpus):
    """Not a preference. Demonstrated rather than asserted, because "NB
    cannot take negative features" is the kind of claim that sits in a
    comment being wrong."""
    from sklearn.naive_bayes import MultinomialNB

    features = spamlib.build_vectorizer(embeddings=True)
    matrix = spamlib.frame_features(features, corpus, fit=True)
    y = corpus["label"].map(spamlib.LABELS).to_numpy()

    assert matrix.min() < 0, "the fake encoder stopped producing signed values"
    with pytest.raises(ValueError):
        MultinomialNB().fit(matrix, y)

    chosen = spamlib.candidate_models(embeddings=True)
    assert not any("Naive Bayes" in name for name in chosen)
    # And the default list still has them, so this is a branch not a removal.
    assert any("Naive Bayes" in name
               for name in spamlib.candidate_models())


def test_every_embedding_candidate_can_actually_be_fitted(encoder, corpus):
    """A list nothing checks is a list with a typo in it."""
    features = spamlib.build_vectorizer(embeddings=True)
    matrix = spamlib.frame_features(features, corpus, fit=True)
    y = corpus["label"].map(spamlib.LABELS).to_numpy()
    for name, estimator in spamlib.candidate_models(embeddings=True).items():
        estimator.fit(matrix, y)
        assert estimator.predict_proba(matrix).shape == (len(y), 2), name


def test_the_regularisation_constant_is_the_one_that_gets_used():
    """EMBEDDING_C exists so the value is stated once. A candidate built
    with the default C would be a second, silent opinion about it."""
    for name, estimator in spamlib.candidate_models(embeddings=True).items():
        assert estimator.C == spamlib.EMBEDDING_C, name


# ------------------------------------------------------------- training

def test_training_with_embeddings_returns_a_thresholded_model(encoder, corpus):
    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    assert isinstance(model, spamlib.Thresholded)
    assert spamlib.wants_raw_text(vectorizer)
    assert 0.0 < model.threshold < 1.0

    verdict = spamlib.predict_message("claim your free prize", model,
                                      vectorizer)
    assert verdict in ("spam", "ham")
    assert 0.0 <= spamlib.spam_probability("hello", model, vectorizer) <= 1.0

    scored = spamlib.metrics(corpus, model, vectorizer)
    assert 0.0 <= scored["f1"] <= 1.0
    assert scored["threshold"] == round(model.threshold, 4)


def test_the_report_names_the_backend_it_used(encoder, corpus, capsys):
    """Two backends produce two tables, and a reader comparing runs needs to
    know which one is on the screen."""
    spamlib.train_and_evaluate(corpus, verbose=True, embeddings=True)
    printed = capsys.readouterr().out
    assert "embedding dimensions" in printed

    spamlib.train_and_evaluate(corpus, verbose=True)
    printed = capsys.readouterr().out
    assert "tf-idf (1, 2)-grams" in printed
    assert "embedding dimensions" not in printed


# -------------------------------------------------------------- artifacts

def test_the_saved_vectorizer_holds_a_path_not_the_weights(encoder, corpus,
                                                           tmp_path):
    """joblib would happily pickle the torch modules and write a
    vectorizer.joblib two orders of magnitude larger than the model it
    describes -- and one that only loads under the same torch version."""
    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    # Force the private attribute to be populated, so dropping it is a real
    # event rather than a no-op on a field that happened to be None.
    vectorizer._encoder = encoder
    blob = pickle.dumps(vectorizer)
    assert len(blob) < 2_000_000, "something big came along for the ride"

    restored = pickle.loads(blob)
    assert restored._encoder is None
    assert restored.model_dir == vectorizer.model_dir
    assert restored.lexical_width == vectorizer.lexical_width


def test_a_reloaded_artifact_classifies_without_being_told_the_backend(
        encoder, corpus, tmp_path):
    """The contract `classify.py` depends on. It has never heard of
    embeddings and must not have to: which backend to use is a property of
    the artifact, not of the command reading it."""
    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    model_path = str(tmp_path / "m.joblib")
    vector_path = str(tmp_path / "v.joblib")
    spamlib.save_model(model, vectorizer, model_path, vector_path)

    loaded_model, loaded_vectorizer = spamlib.load_artifacts(model_path,
                                                             vector_path)
    assert spamlib.wants_raw_text(loaded_vectorizer)

    encoder.seen.clear()
    verdict = spamlib.predict_message("Text WIN to 80086", loaded_model,
                                      loaded_vectorizer)
    assert verdict in ("spam", "ham")
    assert encoder.seen == ["Text WIN to 80086"], (
        "the reloaded artifact stopped routing raw text to the encoder")


def test_feature_names_cover_every_column(encoder, corpus):
    """`app.token_weights` indexes this array by column number and would
    read off the end of it if the two disagreed."""
    features = spamlib.build_vectorizer(embeddings=True)
    matrix = spamlib.frame_features(features, corpus, fit=True)
    names = features.get_feature_names_out()
    assert len(names) == matrix.shape[1]
    assert names[features.lexical_width] == "embedding[0]"


def test_repr_says_which_model_without_dumping_the_weights(encoder, corpus):
    features = spamlib.build_vectorizer(embeddings=True)
    spamlib.frame_features(features, corpus, fit=True)
    text = repr(features)
    assert "SemanticFeatures" in text and str(FAKE_WIDTH) in text


# ----------------------------------------------------- the web app's half

def test_the_explanation_stops_at_the_lexical_columns(encoder, corpus):
    """"Dimension 137 contributed +0.04" is not an explanation. Without the
    boundary the page would list 384 of them and bury the words."""
    import app as web

    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    assert web.explainable_width(vectorizer) == vectorizer.lexical_width

    tokens = web.token_weights("claim your free prize now", model, vectorizer)
    assert tokens, "the explanation came back empty"
    assert not any(t["token"].startswith("embedding[") for t in tokens)


def test_known_tokens_is_not_inflated_by_the_dense_columns(encoder, corpus):
    """An embedding is dense, so all of its columns are non-zero for every
    message. `vec.nnz` would report them as vocabulary hits and the page
    would claim a message of gibberish matched hundreds of known tokens.

    Asserted through `classify`, not just against the helper. The first
    version of this test called `known_tokens` directly, and putting
    `int(vec.nnz)` back into `classify` left it passing -- a test of a
    function nothing was obliged to call.
    """
    import app as web

    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    gibberish = spamlib.vectorize(vectorizer, ["zxqv wbbl"])
    assert gibberish.nnz >= FAKE_WIDTH, "the dense half is not dense"
    assert web.known_tokens(gibberish, vectorizer) < FAKE_WIDTH

    served = web.classify("zxqv wbbl", model, vectorizer)
    assert served["knownTokens"] < FAKE_WIDTH, (
        "the response counts the embedding dimensions as vocabulary hits")

    plain = spamlib.build_vectorizer()
    plain.fit(corpus["clean_text"])
    lexical = spamlib.vectorize(plain, ["claim your free prize"])
    assert web.known_tokens(lexical, plain) == lexical.nnz


def test_the_retrain_button_does_not_change_the_backend(encoder, corpus,
                                                        monkeypatch, tmp_path):
    """Someone installs two gigabytes, trains with --embeddings, opens the
    page, presses Retrain, and gets a tf-idf model back with nothing saying
    so. That is the bug this guards.

    Driven through `ensure_model(force_retrain=True)` rather than only
    against `retrain_with_embeddings`, because the decision being right is
    worth nothing if the retrain path does not ask for it -- which is
    exactly what the first version of this test failed to notice.

    The artifact paths are redirected first. Without that, a retrain here
    would overwrite the project's real model with one trained on an
    eight-dimensional fake encoder, and every later test that loads it would
    be scoring against nonsense.
    """
    import app as web
    import spam_classifier_all_in_one as allinone

    monkeypatch.setattr(allinone, "MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setattr(allinone, "VECTORIZER_PATH", str(tmp_path / "v.joblib"))

    _model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                    embeddings=True)
    monkeypatch.setitem(web._state, "vectorizer", vectorizer)
    assert web.retrain_with_embeddings() is True

    try:
        _retrained, rebuilt = web.ensure_model(force_retrain=True)
        assert spamlib.wants_raw_text(rebuilt), (
            "retraining silently dropped the embedding backend")
    finally:
        web.reset()

    monkeypatch.setitem(web._state, "vectorizer", spamlib.build_vectorizer())
    assert web.retrain_with_embeddings() is False


def test_the_app_scores_a_raw_text_backend_on_raw_text(encoder, corpus):
    """`evaluate` used to split on the normalised column unconditionally."""
    import app as web

    model, vectorizer = spamlib.train_and_evaluate(corpus, verbose=False,
                                                   embeddings=True)
    encoder.seen.clear()
    scored = web.evaluate(corpus, model, vectorizer)
    assert scored["testSize"] > 0
    assert all(seen in set(corpus["text"]) for seen in encoder.seen)


# --------------------------------------------------------------- the CLIs

def test_both_clis_offer_the_flag():
    for script in ("spam_classifier_all_in_one.py", "train_spam_classifier.py"):
        done = subprocess.run([sys.executable, os.path.join(ROOT, script),
                               "--help"],
                              capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-400:]
        assert "--embeddings" in done.stdout, script


def test_the_flag_prints_the_fix_rather_than_a_traceback(monkeypatch, capsys):
    """Asking for a backend you have not installed should answer with the
    pip command. Both entry points, one message, because it comes from
    `spamlib.embedding_backend_problem`."""
    monkeypatch.setattr(embeddings.importlib.util, "find_spec",
                        lambda name: None)
    import spam_classifier_all_in_one as allinone

    for module in (trainer, allinone):
        assert module.main(["--embeddings"]) == 1
        printed = capsys.readouterr()
        assert "requirements-embeddings.txt" in printed.err, module.__name__


def test_classifying_takes_no_backend_flag():
    """Deliberate: the artifact decides. A flag here could only contradict
    the file on disk."""
    source = io.open(os.path.join(ROOT, "spam_classifier_all_in_one.py"),
                     encoding="utf-8").read()
    body = source[source.index("def main("):]
    assert "args.classify" in body
    # The availability check must sit after the --classify early return, or
    # reading a saved model would demand an install it does not need.
    assert body.index("args.classify") < body.index("args.embeddings")


# ------------------------------------------------- only with the real model

needs_model = pytest.mark.skipif(
    not embeddings.available(),
    reason="the sentence encoder is not installed; see "
           "requirements-embeddings.txt")


@needs_model
def test_the_real_encoder_produces_the_documented_width():
    """The one thing a fake cannot check: that the files on disk load, and
    load as the model they claim to be."""
    embeddings.reset()
    try:
        vectors = embeddings.encode(MESSAGES)
        assert vectors.shape == (len(MESSAGES), embeddings.MINILM_DIMENSION)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "not unit-normalised"
    finally:
        embeddings.reset()


@needs_model
def test_the_real_encoder_is_a_semantic_space_not_384_arbitrary_numbers():
    """What the backend is for, stated as the weakest claim that would fail
    if the wrong weights were loaded: a paraphrase is closer than a
    different subject. Not a score -- `tools/compare_backends.py` measures
    scores -- just evidence that the space means something.
    """
    embeddings.reset()
    try:
        vectors = embeddings.encode([
            "Your parcel could not be delivered, pay the fee here",
            "We could not deliver your package; settle the charge now",
            "See you at football training on Thursday",
        ])
    finally:
        embeddings.reset()

    paraphrase = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert paraphrase > unrelated + 0.2, (paraphrase, unrelated)
