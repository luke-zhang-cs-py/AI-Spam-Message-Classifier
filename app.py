"""
app.py
-------
Web front-end for the spam classifier.

    pip install -r requirements.txt
    python app.py
Then open:  http://127.0.0.1:5002

The model itself is untouched — this imports spam_classifier_all_in_one and
calls the same functions the CLI does. What the browser adds over
`--classify` is an explanation: which individual words pushed the verdict
towards spam or ham, and by how much.

Endpoints
---------
GET  /                 the page
POST /api/classify      body {message} -> verdict, confidence, token weights
POST /api/batch         body {messages: [...]} -> one verdict per line
POST /api/retrain       retrain from data/ and reload, same backend
GET  /api/model         which model is loaded, and its evaluation scores

Binds to 127.0.0.1. Nothing here is authenticated, and messages people paste
in to test are exactly the kind of thing you would rather not expose.
"""

import threading

import numpy as np
from flask import Flask, jsonify, render_template, request

import spam_classifier_all_in_one as clf

app = Flask(__name__)

# One message big enough to be a real one and small enough not to be a
# weapon. Cleaning and vectorising is linear in the text, and a batch of
# 200 uncapped messages was several hundred megabytes of work per request.
MAX_MESSAGE_CHARS = 20_000
MAX_BATCH = 200

_state = {"model": None, "vectorizer": None, "name": None,
          "metrics": None, "datasetSize": None}
_lock = threading.Lock()


def reset():
    """Forget the cached model.

    Every other module cache in this family of projects has one of these --
    fxrates.reset, schedule.reset, realtime.reset_cache -- and this one did
    not. It matters for the retrain endpoint: force_retrain=True replaces the
    process-wide model, so a test that exercises it leaves every later test
    running against a different object than it started with. Harmless while
    the training is deterministic, and a real order-dependence the moment it
    stops being.
    """
    with _lock:
        _state.update(model=None, vectorizer=None, name=None,
                      metrics=None, datasetSize=None)


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------

def model_display_name(model):
    return type(model).__name__ if model is not None else None


def retrain_with_embeddings():
    """Whether a retrain should keep the optional embedding backend.

    The retrain button must not change which features the model uses. Left
    to the default it would: someone trains with `--embeddings`, opens the
    page, presses Retrain, and gets a tf-idf model back with no indication
    that the thing they installed two gigabytes for is no longer in play.

    So the answer is read from what is currently loaded, and from the saved
    artifact if nothing is. Called inside the lock, and deliberately before
    `_state` is overwritten.
    """
    current = _state["vectorizer"]
    if current is None:
        current = clf.load_artifacts()[1]
    return current is not None and clf.wants_raw_text(current)


def _ensure_model_locked(force_retrain=False):
    """The actual work of `ensure_model`. Caller must already hold `_lock`.

    Split out so a route that needs both the (model, vectorizer) pair *and*
    a consistent read of `_state` -- `api_model`, `api_retrain` -- can do
    both inside one `with _lock:` block instead of calling `ensure_model()`
    (which releases the lock on return) and then reacquiring it to read
    `_state`. That gap let a concurrent `/api/retrain` swap `_state` out
    from under a request in the middle of it, so the name/metrics/
    datasetSize in one response could come from two different trainings.
    """
    if _state["model"] is not None and not force_retrain:
        return _state["model"], _state["vectorizer"]

    model, vectorizer = (None, None) if force_retrain else clf.load_artifacts()

    df = clf.load_data()
    if model is None or vectorizer is None:
        model, vectorizer = clf.train_and_evaluate(
            df, embeddings=retrain_with_embeddings())
        clf.save_model(model, vectorizer)

    # Scored whether it was just trained or loaded from disk. This used to
    # run only on the training path: the .joblib artifacts are gitignored,
    # so the first run after a clone trained a model and showed its scores,
    # and every restart afterwards loaded that model and showed nothing --
    # a metric that disappears is worse than one that was never there. The
    # page even had a line explaining it away as inherent ("a model loaded
    # from disk carries no metrics with it"). It is not: evaluate() runs
    # spamlib's out-of-fold cross-validation against this dataset, cloning
    # and refitting the estimator per fold rather than scoring the
    # already-fully-fit model on rows it was trained on. The one
    # assumption is that the artifacts were trained on this dataset; swap
    # dataset.csv without retraining and the reported numbers describe how
    # this kind of model performs on the new data, not the loaded model
    # itself, which is what the retrain button is for.
    _state.update(model=model, vectorizer=vectorizer,
                  name=model_display_name(model),
                  metrics=evaluate(df, model, vectorizer),
                  datasetSize=int(len(df)))
    return model, vectorizer


def ensure_model(force_retrain=False):
    """Load the saved artifacts, training them first if they are missing.

    Held for the whole operation rather than released between the check and
    the work. Two first requests arriving together both used to see an empty
    cache, and both went off and trained a model.
    """
    with _lock:
        return _ensure_model_locked(force_retrain)


def evaluate(df, model, vectorizer):
    """Genuine held-out metrics for display in the UI.

    This used to carve its own train_test_split out of `df` and score the
    model on that slice -- but train_and_evaluate() fits the final estimator
    on the *entire* dataframe ("fit on everything", by its own docstring), so
    every row in that "test" slice had already been trained on. The numbers
    were in-sample and optimistic, not held-out.

    Delegates to spamlib.metrics() instead, which never scores a row with a
    model that was fit on it: it clones the chosen estimator and runs it
    through StratifiedKFold cross-validation, so each row is only ever
    predicted by a fold that did not train on it.
    """
    scored = clf.metrics(df, model, vectorizer)
    return {
        "accuracy": scored["accuracy"],
        "precision": scored["precision"],
        "recall": scored["recall"],
        "f1": scored["f1"],
        "confusion": scored["confusionMatrix"],
        "testSize": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def token_weights(message, model, vectorizer, limit=12):
    """Per-word contribution to the spam-vs-ham decision.

    Both classifiers reduce to a linear score over TF-IDF features, so a
    token's contribution is its TF-IDF value times the model's weight for it.
    The two models store that weight differently:

        LogisticRegression  coef_ is already the log-odds weight.
        MultinomialNB       feature_log_prob_ holds log P(word | class); the
                            difference between the spam and ham rows is the
                            equivalent log-ratio weight.

    Positive pushes towards spam, negative towards ham. Only tokens actually
    present in the message have a non-zero TF-IDF, so this returns just the
    words that mattered.

    With the optional embedding backend loaded the matrix has 384 further
    columns, and they are left out on purpose: "dimension 137 contributed
    +0.04" is not an explanation, it is a number with a label on it. The
    lexical half is the half that can be read, so the story stops at
    `lexical_width` -- see `explainable_width` below.
    """
    vec = clf.vectorize(vectorizer, [message])
    if vec.nnz == 0:
        return []

    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_).ravel()
    elif hasattr(model, "feature_log_prob_"):
        lp = np.asarray(model.feature_log_prob_)
        weights = lp[1] - lp[0]
    else:
        return []

    names = vectorizer.get_feature_names_out()
    limit_column = explainable_width(vectorizer)
    row = vec.tocoo()
    contributions = [
        {"token": str(names[col]),
         "weight": round(float(val * weights[col]), 4),
         "tfidf": round(float(val), 4)}
        for col, val in zip(row.col, row.data)
        if col < limit_column
    ]
    contributions.sort(key=lambda c: -abs(c["weight"]))
    return contributions[:limit]


def explainable_width(vectorizer):
    """How many leading columns correspond to a word a reader can see.

    For tf-idf that is all of them. For the combined backend it is
    `lexical_width`, because the embedding columns carry no per-word
    meaning. Read by attribute so app.py does not import the optional
    module -- the whole point of which is not being imported.

    The test is `is None`, not truthiness. `lexical_width` is 0 on a
    SemanticFeatures built with no lexical half at all, and the earlier
    `if width:` read that 0 as "this is a plain vectoriser, explain
    everything" -- so an embeddings-only backend reported all 384 dimensions
    as explainable words, which is the exact opposite of what the boundary
    is for. Absent attribute means tf-idf; present-and-zero means there is
    genuinely nothing to explain.
    """
    width = getattr(vectorizer, "lexical_width", None)
    if width is None:
        return len(vectorizer.get_feature_names_out())
    return int(width)


def classify(message, model, vectorizer):
    # `cleaned` is still shown in the response: it is how the page explains
    # what normalisation did to the message. It is no longer what gets
    # vectorised, though -- clf.vectorize asks the backend what form it
    # wants and normalises only if the answer is "cleaned".
    cleaned = clf.clean_text(message)
    vec = clf.vectorize(vectorizer, [message])
    pred = int(model.predict(vec)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        # By class order, not by label value -- see predict_message in
        # spam_classifier_all_in_one for why they are not the same thing.
        index = list(model.classes_).index(pred)
        confidence = round(float(model.predict_proba(vec)[0][index]), 4)

    return {
        "message": message,
        "cleaned": cleaned,
        "label": "spam" if pred == 1 else "ham",
        "confidence": confidence,
        "knownTokens": known_tokens(vec, vectorizer),
        "tokens": token_weights(message, model, vectorizer),
    }


def known_tokens(vec, vectorizer):
    """How many vocabulary terms the message actually hit.

    `vec.nnz` was this number for as long as every column was a term. With
    the embedding backend it is not: an embedding is dense, so all 384 of
    its columns are non-zero for every message, and the page would report
    "384 known tokens" for a message of pure gibberish. Counted over the
    lexical columns only, which is what the label on the page means.

    Defers to `explainable_width` rather than repeating its lookup. The two
    functions had the same `getattr(..., "lexical_width", None)` and the
    same truthiness test written out separately, which is how they would
    have come to disagree about where the boundary is -- and they were
    already wrong in the same way, which is the other thing duplicated
    logic gets you.
    """
    width = explainable_width(vectorizer)
    if width >= vec.shape[1]:
        return int(vec.nnz)
    return int((vec.tocoo().col < width).sum())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model")
def api_model():
    """What is loaded and how well it scores.

    datasetSize comes from the cached state rather than re-reading and
    re-parsing the CSV, which is what this did on every single request.

    The ensure-then-read used to be two separate critical sections --
    `ensure_model()` (lock released on return), then a fresh `with _lock:`
    to read `_state`. A `/api/retrain` landing in the gap between them could
    swap `_state` out from under this request, so the name/metrics/
    datasetSize in one response were not guaranteed to all come from the
    same training. One acquisition now covers both.
    """
    with _lock:
        _ensure_model_locked()
        return jsonify({"name": _state["name"], "metrics": _state["metrics"],
                        "datasetSize": _state["datasetSize"]})


@app.route("/api/classify", methods=["POST"])
def api_classify():
    body = request.get_json(force=True, silent=True) or {}
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"ok": False, "error": "Type a message first."}), 400
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"ok": False,
                        "error": f"Messages are capped at {MAX_MESSAGE_CHARS:,} "
                                 f"characters."}), 400

    model, vectorizer = ensure_model()
    return jsonify({"ok": True, **classify(message, model, vectorizer)})


def clean_batch(raw):
    """The usable messages out of a request body, or a ValueError saying why.

    Pulled out of the handler, which had grown six validate-and-return pairs
    wrapped around three lines of actual work. Each rule is here because
    something got through without it:

    * not a list -- a bare string was iterated character by character, so
      {"messages": "free money"} came back 200 with a nine-message summary;
    * not all strings -- an int raised AttributeError inside the
      comprehension and came back as a 500;
    * length -- cleaning and vectorising is linear in the text, and 200
      uncapped messages is hundreds of megabytes of work in one request.
    """
    if not isinstance(raw, list):
        raise ValueError("'messages' must be a list of strings.")
    if any(not isinstance(m, str) for m in raw if m is not None):
        raise ValueError("Every message must be a string.")

    messages = [m.strip() for m in raw if isinstance(m, str) and m.strip()]
    if not messages:
        raise ValueError("No messages to classify.")
    if len(messages) > MAX_BATCH:
        raise ValueError(f"Cap is {MAX_BATCH} messages at a time.")
    if any(len(m) > MAX_MESSAGE_CHARS for m in messages):
        raise ValueError(f"Messages are capped at {MAX_MESSAGE_CHARS:,} characters.")
    return messages


@app.route("/api/batch", methods=["POST"])
def api_batch():
    body = request.get_json(force=True, silent=True) or {}
    try:
        messages = clean_batch(body.get("messages"))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    model, vectorizer = ensure_model()
    results = [classify(m, model, vectorizer) for m in messages]
    spam = sum(1 for r in results if r["label"] == "spam")
    return jsonify({"ok": True, "results": results,
                    "summary": {"total": len(results), "spam": spam,
                                "ham": len(results) - spam}})


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Retrain and report the result from one lock acquisition.

    Same reasoning as `api_model`: a separate `ensure_model()` call followed
    by its own `with _lock:` to read `_state` back left a window for another
    request to interleave and change what got reported.
    """
    with _lock:
        _ensure_model_locked(force_retrain=True)
        return jsonify({"ok": True, "name": _state["name"],
                        "metrics": _state["metrics"]})


if __name__ == "__main__":
    ensure_model()
    with _lock:
        print(f"Model ready: {_state['name']}")
    print("Open http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, threaded=True, use_reloader=False)
