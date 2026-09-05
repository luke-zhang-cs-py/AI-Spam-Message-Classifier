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
POST /api/retrain       retrain from the embedded dataset and reload
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


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------

def model_display_name(model):
    return type(model).__name__ if model is not None else None


def ensure_model(force_retrain=False):
    """Load the saved artifacts, training them first if they are missing.

    Held for the whole operation rather than released between the check and
    the work. Two first requests arriving together both used to see an empty
    cache, and both went off and trained a model.
    """
    with _lock:
        if _state["model"] is not None and not force_retrain:
            return _state["model"], _state["vectorizer"]

        model, vectorizer = (None, None) if force_retrain else clf.load_artifacts()

        df = clf.load_data()
        if model is None or vectorizer is None:
            model, vectorizer = clf.train_and_evaluate(df)
            clf.save_model(model, vectorizer)

        # Scored whether it was just trained or loaded from disk. This used to
        # run only on the training path: the .joblib artifacts are gitignored,
        # so the first run after a clone trained a model and showed its scores,
        # and every restart afterwards loaded that model and showed nothing --
        # a metric that disappears is worse than one that was never there. The
        # page even had a line explaining it away as inherent ("a model loaded
        # from disk carries no metrics with it"). It is not: evaluate()
        # re-scores against the same fixed split the trainer used. The one
        # assumption is that the artifacts were trained on this dataset; swap
        # dataset.csv without retraining and the split moves under them, which
        # is what the retrain button is for.
        _state.update(model=model, vectorizer=vectorizer,
                      name=model_display_name(model),
                      metrics=evaluate(df, model, vectorizer),
                      datasetSize=int(len(df)))
    return model, vectorizer


def evaluate(df, model, vectorizer):
    """Re-score on the same split the script uses, for display in the UI."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})
    # clf's constants, not a second copy of the numbers: the quarter this
    # scores against is only held-out data if it is the same quarter the
    # trainer held out.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=clf.TEST_SIZE, random_state=clf.RANDOM_STATE, stratify=y)

    preds = model.predict(vectorizer.transform(X_test))
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 3),
        "confusion": confusion_matrix(y_test, preds).tolist(),
        "testSize": int(len(y_test)),
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
    """
    cleaned = clf.clean_text(message)
    vec = vectorizer.transform([cleaned])
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
    row = vec.tocoo()
    contributions = [
        {"token": str(names[col]),
         "weight": round(float(val * weights[col]), 4),
         "tfidf": round(float(val), 4)}
        for col, val in zip(row.col, row.data)
    ]
    contributions.sort(key=lambda c: -abs(c["weight"]))
    return contributions[:limit]


def classify(message, model, vectorizer):
    cleaned = clf.clean_text(message)
    vec = vectorizer.transform([cleaned])
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
        "knownTokens": int(vec.nnz),
        "tokens": token_weights(message, model, vectorizer),
    }


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
    """
    ensure_model()
    with _lock:
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
    model, _ = ensure_model(force_retrain=True)
    with _lock:
        return jsonify({"ok": True, "name": _state["name"],
                        "metrics": _state["metrics"]})


if __name__ == "__main__":
    ensure_model()
    with _lock:
        print(f"Model ready: {_state['name']}")
    print("Open http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, threaded=True, use_reloader=False)
