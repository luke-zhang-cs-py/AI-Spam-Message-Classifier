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

from train_spam_classifier import MODEL_PATH, VECTORIZER_PATH, predict_message


def load_artifacts():
    """Returns (model, vectorizer), or (None, None) if they are not on disk.

    It used to print and call sys.exit(1) from in here. A function whose job
    is to load two files should not be able to end the process -- it made
    this untestable, and it meant that importing `load_artifacts` from this
    module rather than from spam_classifier_all_in_one silently handed you a
    function with the power to terminate your program. Two functions with
    one name and opposite contracts is the trap. Deciding what to do about a
    missing model is main()'s job, below.
    """
    try:
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        return None, None


def classify(message: str, model, vectorizer) -> str:
    """The command line's rendering of the shared verdict.

    This used to reimplement predict_message -- same transform, same
    predict, same probability lookup -- and returned "SPAM (69.1%
    confidence)" where the original returns "spam (confidence: 69.10%)". One
    decision, three entry points, three spellings of the answer. The logic is
    imported now; only the capitalisation is this file's own.
    """
    verdict = predict_message(message, model, vectorizer)
    label, _, rest = verdict.partition(" ")
    return f"{label.upper()} {rest}".strip()


def main():
    model, vectorizer = load_artifacts()
    if model is None:
        print("Model not found. Run `python train_spam_classifier.py` first.")
        return 1

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(classify(message, model, vectorizer))
        return 0

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
