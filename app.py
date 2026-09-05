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

import os
import threading

from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # MODEL_PATH and VECTORIZER_PATH are relative

import numpy as np                              # noqa: E402
import spam_classifier_all_in_one as clf        # noqa: E402

app = Flask(__name__)

_state = {"model": None, "vectorizer": None, "name": None, "metrics": None}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------

def model_display_name(model):
    return type(model).__name__ if model is not None else None


def ensure_model(force_retrain=False):
    """Load the saved artifacts, training them first if they are missing."""
    with _lock:
        if _state["model"] is not None and not force_retrain:
            return _state["model"], _state["vectorizer"]

    model, vectorizer = (None, None) if force_retrain else clf.load_artifacts()

    metrics = None
    if model is None or vectorizer is None:
        df = clf.load_data()
        model, vectorizer = clf.train_and_evaluate(df)
        clf.save_model(model, vectorizer)
        metrics = evaluate(df, model, vectorizer)

    with _lock:
        _state.update(model=model, vectorizer=vectorizer,
                      name=model_display_name(model))
        if metrics:
            _state["metrics"] = metrics
    return model, vectorizer


def evaluate(df, model, vectorizer):
    """Re-score on the same split the script uses, for display in the UI."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

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
        confidence = round(float(model.predict_proba(vec)[0][pred]), 4)

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
    model, _ = ensure_model()
    with _lock:
        return jsonify({
            "name": _state["name"],
            "metrics": _state["metrics"],
            "datasetSize": int(len(clf.load_data())),
        })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    body = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Type a message first."}), 400

    model, vectorizer = ensure_model()
    return jsonify({"ok": True, **classify(message, model, vectorizer)})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    body = request.get_json(force=True, silent=True) or {}
    messages = [m.strip() for m in (body.get("messages") or []) if m and m.strip()]
    if not messages:
        return jsonify({"ok": False, "error": "No messages to classify."}), 400
    if len(messages) > 200:
        return jsonify({"ok": False, "error": "Cap is 200 messages at a time."}), 400

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
