"""Tests for the web front-end and the model plumbing behind it.

The repository had none. Every test that names a bug describes one that was
really here.

The model itself is not re-trained per test -- ensure_model caches it -- so
this runs in seconds rather than minutes.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as web                                  # noqa: E402
import spam_classifier_all_in_one as clf           # noqa: E402

SPAM = "WINNER! Claim your FREE prize now, call 09061701461 to collect."
HAM = "Hey, are we still on for coffee at 6?"


@pytest.fixture(scope="module")
def client():
    web.app.config["TESTING"] = True
    return web.app.test_client()


# ------------------------------------------------------------------- paths

def test_artifacts_are_found_from_any_directory(tmp_path, monkeypatch):
    """These were bare relative filenames, so the model landed wherever you
    happened to be standing -- and app.py compensated by calling os.chdir()
    at import time, which moves the whole process.

    The paths are asserted before anything is trained, because that is the
    real subject: absolute, and derived from the module rather than the cwd.
    It then trains *while standing in tmp_path*, which is the stronger form
    of the same claim -- the artifacts have to land beside the module, not
    beside wherever the process happens to be.

    It used to assert os.path.exists() on a model nobody had built. The
    .joblib files are gitignored, so that held only on a machine where an
    earlier run had left one behind, and failed in every fresh clone.
    """
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(clf.MODEL_PATH)
    assert os.path.isabs(clf.VECTORIZER_PATH)
    assert os.path.dirname(clf.MODEL_PATH) == ROOT

    web.ensure_model()

    assert os.path.exists(clf.MODEL_PATH)
    assert not list(tmp_path.glob("*.joblib")), "artifacts written to the cwd"
    model, vectorizer = clf.load_artifacts()
    assert model is not None and vectorizer is not None


def test_importing_the_app_does_not_move_the_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(web)
    assert os.getcwd() == str(tmp_path)


# ------------------------------------------------------------------- model

def test_the_shipped_model_reports_its_scores(client):
    """The bug: evaluate() ran only on the training path.

    The artifacts are gitignored, so the first run after a clone trains and
    shows scores -- and every restart after that loads them from disk and
    showed nothing, which is a stranger thing to watch than never showing
    them at all. The page had a line explaining it away as inherent. It is
    not: the evaluation runs against a fixed split, so a loaded model can be
    scored exactly as well as a fresh one."""
    body = client.get("/api/model").get_json()
    assert body["name"]
    metrics = body["metrics"]
    assert metrics is not None, "a loaded model can be scored too"
    for key in ("accuracy", "precision", "recall", "f1", "testSize"):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["testSize"] > 0


def test_the_model_endpoint_does_not_reparse_the_dataset(client, monkeypatch):
    """It called load_data() on every single request, to report a number that
    does not change."""
    calls = []
    real = clf.load_data
    monkeypatch.setattr(clf, "load_data", lambda: calls.append(1) or real())
    for _ in range(5):
        client.get("/api/model")
    assert calls == []


def test_dataset_size_is_reported(client):
    body = client.get("/api/model").get_json()
    assert body["datasetSize"] == len(clf.load_data())


# ---------------------------------------------------------------- classify

def test_spam_is_called_spam(client):
    body = client.post("/api/classify", json={"message": SPAM}).get_json()
    assert body["ok"] and body["label"] == "spam"
    assert 0.0 <= body["confidence"] <= 1.0


def test_ham_is_called_ham(client):
    body = client.post("/api/classify", json={"message": HAM}).get_json()
    assert body["ok"] and body["label"] == "ham"


def test_the_explanation_names_words_from_the_message(client):
    """The whole point of the web version over the CLI."""
    body = client.post("/api/classify", json={"message": SPAM}).get_json()
    assert body["tokens"], "no tokens explained"
    # The vectorizer is ngram_range=(1, 2) over English stop words, so a
    # feature can be a bigram of two words that were not adjacent in the
    # original -- "claim free" out of "claim your free". Checking each word
    # rather than the phrase is the check that is actually true.
    words = set(body["cleaned"].split())
    for token in body["tokens"]:
        assert set(token["token"].split()) <= words, token["token"]
        assert "weight" in token and "tfidf" in token
    weights = [abs(t["weight"]) for t in body["tokens"]]
    assert weights == sorted(weights, reverse=True), "most influential first"


def test_a_message_of_pure_gibberish_explains_nothing(client):
    """No known tokens means no evidence, and the response should say so
    rather than inventing an explanation."""
    body = client.post("/api/classify", json={"message": "zzzz qqqq xxxx"}).get_json()
    assert body["ok"]
    assert body["knownTokens"] == 0
    assert body["tokens"] == []


@pytest.mark.parametrize("body,why", [
    ({}, "no message"),
    ({"message": ""}, "empty"),
    ({"message": "   "}, "whitespace"),
    ({"message": None}, "null"),
    ({"message": 12345}, "not a string"),
    ({"message": ["a", "b"]}, "a list"),
])
def test_an_unusable_message_is_400_not_500(client, body, why):
    assert client.post("/api/classify", json=body).status_code == 400, why


def test_an_enormous_message_is_refused(client):
    """Cleaning and vectorising is linear in the text. Uncapped, one request
    was megabytes of work, and a batch multiplied it by two hundred."""
    res = client.post("/api/classify", json={"message": "free " * 100000})
    assert res.status_code == 400
    assert "capped" in res.get_json()["error"]


# ------------------------------------------------------------------- batch

def test_a_batch_classifies_every_line(client):
    body = client.post("/api/batch", json={"messages": [SPAM, HAM]}).get_json()
    assert body["ok"]
    assert body["summary"] == {"total": 2, "spam": 1, "ham": 1}
    assert [r["label"] for r in body["results"]] == ["spam", "ham"]


@pytest.mark.parametrize("messages,why", [
    ([1, 2, 3], "ints raised AttributeError inside the comprehension: a 500"),
    ([["a"]], "a nested list did the same"),
    ("free money", "a bare string was iterated character by character"),
    ({"a": "b"}, "a dict was iterated over its keys"),
    (None, "missing"),
    (42, "not a list at all"),
])
def test_a_malformed_batch_is_400_not_500_or_nonsense(client, messages, why):
    res = client.post("/api/batch", json={"messages": messages})
    assert res.status_code == 400, why


def test_a_bare_string_is_not_nine_one_letter_messages(client):
    """It returned 200 and a cheerful nine-message summary."""
    res = client.post("/api/batch", json={"messages": "free money"})
    assert res.status_code == 400


def test_nulls_in_a_batch_are_skipped_not_fatal(client):
    body = client.post("/api/batch", json={"messages": [SPAM, None, HAM]}).get_json()
    assert body["summary"]["total"] == 2


def test_the_batch_cap_is_enforced(client):
    res = client.post("/api/batch", json={"messages": [HAM] * (web.MAX_BATCH + 1)})
    assert res.status_code == 400


def test_an_empty_batch_is_refused(client):
    assert client.post("/api/batch", json={"messages": []}).status_code == 400
    assert client.post("/api/batch", json={"messages": ["  ", ""]}).status_code == 400


# -------------------------------------------------------------- the model

def test_cleaning_is_what_the_cli_does(client):
    """The web front-end must not quietly preprocess differently from the
    command line, or the two disagree about the same message."""
    body = client.post("/api/classify", json={"message": SPAM}).get_json()
    assert body["cleaned"] == clf.clean_text(SPAM)


def test_the_page_loads(client):
    assert client.get("/").status_code == 200
