"""
AI Spam Message Classifier
---------------------------
Trains a small machine learning model to classify text messages as
"spam" or "ham" (not spam), using scikit-learn.

Pipeline:
    1. Load labeled message data (dataset.csv: columns = label, text)
    2. Clean/preprocess the text
    3. Convert text to numeric features with TF-IDF
    4. Train a Multinomial Naive Bayes classifier (fast, well-suited to text)
    5. Evaluate accuracy, precision, recall, F1, and confusion matrix
    6. Save the trained model + vectorizer to disk for reuse
    7. Provide a predict_message() helper to classify new messages

Usage:
    python train_spam_classifier.py
"""

import re
import string
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# The training recipe comes from the canonical module rather than being
# restated here. These were four literals -- 0.25, 42, (1, 2) and 1 -- that
# had to match the ones app.py reads by name in order for its held-out
# quarter to be genuinely held out. They did match, but only by coincidence:
# nothing compared them, and the comment over there described the duplication
# in the past tense while this half of it was still sitting here.
from spam_classifier_all_in_one import (  # noqa: E402
    MAX_ITER, MIN_DF, NGRAM_RANGE, RANDOM_STATE, STOP_WORDS, TEST_SIZE,
)

# Resolved against this file, not the working directory. They used to be
# bare relative names, which meant the artifacts landed wherever you happened
# to be standing when you ran the script -- and app.py compensated by calling
# os.chdir() at import time, a process-wide side effect of an import.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "dataset.csv")
MODEL_PATH = os.path.join(_HERE, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(_HERE, "vectorizer.joblib")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation noise from a message."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                        # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def train_and_evaluate(df: pd.DataFrame):
    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF,
                                 stop_words=STOP_WORDS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=MAX_ITER),
    }

    best_model = None
    best_score = -1
    best_name = None

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 score : {f1:.3f}")
        print("Confusion matrix [[TN FP]\n                  [FN TP]]:")
        print(confusion_matrix(y_test, preds))

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    print("\n" + "=" * 60)
    print(f"Best model: {best_name} (F1 = {best_score:.3f})")
    print("=" * 60)
    print("\nDetailed report for best model:")
    print(classification_report(
        y_test, best_model.predict(X_test_vec), target_names=["ham", "spam"]
    ))

    return best_model, vectorizer


def save_model(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved trained model to '{MODEL_PATH}'")
    print(f"Saved vectorizer to '{VECTORIZER_PATH}'")


def predict_message(message: str, model, vectorizer) -> str:
    """Classify a single new message as 'spam' or 'ham'."""
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred] if hasattr(model, "predict_proba") else None
    label = "spam" if pred == 1 else "ham"
    confidence = f" (confidence: {prob:.2%})" if prob is not None else ""
    return f"{label}{confidence}"


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages "
          f"({(df['label'] == 'spam').sum()} spam, {(df['label'] == 'ham').sum()} ham)\n")

    model, vectorizer = train_and_evaluate(df)
    save_model(model, vectorizer)

    print("\n" + "=" * 60)
    print("TRY IT OUT: classifying new sample messages")
    print("=" * 60)
    sample_messages = [
        "Congratulations, you have won a free ticket! Click here to claim now",
        "Hey, are you free for dinner tonight?",
        "URGENT: verify your bank account now or it will be suspended",
        "Can you send me the meeting notes from earlier?",
    ]
    for msg in sample_messages:
        result = predict_message(msg, model, vectorizer)
        print(f"  \"{msg}\"\n   --> {result}\n")


if __name__ == "__main__":
    main()
