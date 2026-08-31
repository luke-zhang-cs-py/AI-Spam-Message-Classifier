"""
Command-line spam classifier.

Loads the model + vectorizer saved by train_spam_classifier.py and
classifies message(s) you provide.

Usage:
    python classify.py "Congratulations, you've won a free prize!"
    python classify.py   # no argument -> interactive mode, type messages, Ctrl+C to quit
"""

import sys
import joblib
from train_spam_classifier import clean_text, MODEL_PATH, VECTORIZER_PATH


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        print("Model not found. Run `python train_spam_classifier.py` first.")
        sys.exit(1)
    return model, vectorizer


def classify(message: str, model, vectorizer) -> str:
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred] if hasattr(model, "predict_proba") else None
    label = "SPAM" if pred == 1 else "HAM"
    confidence = f" ({prob:.1%} confidence)" if prob is not None else ""
    return f"{label}{confidence}"


def main():
    model, vectorizer = load_artifacts()

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(classify(message, model, vectorizer))
        return

    print("Spam classifier — type a message and press Enter (Ctrl+C to quit)\n")
    try:
        while True:
            message = input("> ")
            if message.strip():
                print(" ", classify(message, model, vectorizer))
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
