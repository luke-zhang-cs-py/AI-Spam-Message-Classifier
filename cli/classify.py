"""
Command-line spam classifier.

Loads the model + vectorizer saved by train_spam_classifier.py and
classifies message(s) you provide.

Usage:
    python -m cli.classify "Congratulations, you've won a free prize!"
    python -m cli.classify   # no argument -> interactive, Ctrl+C to quit
"""

import sys

from pipeline import spamlib
from cli.train_spam_classifier import MODEL_PATH, VECTORIZER_PATH, predict_message


def load_artifacts():
    """Returns (model, vectorizer), or (None, None) if they cannot be loaded.

    It used to print and call sys.exit(1) from in here. A function whose job
    is to load two files should not be able to end the process -- it made
    this untestable, and it meant that importing `load_artifacts` from this
    module rather than from spam_classifier_all_in_one silently handed you a
    function with the power to terminate your program. Deciding what to do
    about a missing model is main()'s job, below.

    It then kept its own `joblib.load` pair, catching FileNotFoundError and
    nothing else, beside the shared loader that catches every failure -- so
    an artifact written by another scikit-learn, or truncated by an
    interrupted save, was "Model not found" to the trainer and a traceback
    here. It defers to the shared loader now, with this module's paths
    (which the tests monkeypatch).
    """
    return spamlib.load_artifacts(MODEL_PATH, VECTORIZER_PATH)


def classify(message: str, model, vectorizer) -> str:
    """The command line's rendering of the shared verdict: "SPAM" or "HAM".

    This used to reimplement predict_message -- same transform, same
    predict, same probability lookup -- with a "SPAM (69.1% confidence)"
    format of its own invention. The decision is imported now, so this file
    only supplies the capitals.
    """
    return predict_message(message, model, vectorizer).upper()


def run_interactive(model, vectorizer, read=None):
    """Classify messages typed at a prompt until Ctrl-C or EOF.

    `read` is a parameter for the same reason it is one in
    `spam_classifier_all_in_one.run_interactive`: this loop used to call
    `input()` directly, which put the whole interactive half of an
    interactive tool out of reach of any test. It was the largest uncovered
    block in the project and the one most likely to be broken without
    anybody noticing, because the only way to run it was by hand.

    It defaults to None and resolves to `input` here rather than defaulting
    to `read=input` in the signature. A default argument is evaluated once,
    when the module is imported, so `read=input` captures the builtin as it
    was at import time and quietly ignores any later replacement of it --
    including the one a test harness installs.

    EOFError is caught alongside KeyboardInterrupt. A piped or closed stdin
    raises the former, and the version that caught only the latter ended a
    `classify.py < /dev/null` with a traceback.

    The two loops in this project stay separate on purpose: the all-in-one
    prints the probability and the cut, this one prints the shared verdict
    in upper case. Same decision, two deliberate renderings.
    """
    reader = input if read is None else read
    print("Spam classifier — type a message and press Enter (Ctrl+C to quit)\n")
    while True:
        try:
            message = reader("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if message.strip():
            print(" ", classify(message, model, vectorizer))


def main(argv=None, read=None):
    """`argv` excludes the program name, like `sys.argv[1:]`."""
    arguments = sys.argv[1:] if argv is None else list(argv)

    model, vectorizer = load_artifacts()
    if model is None:
        print("Model not found. Run `python -m cli.train_spam_classifier` "
              "first.")
        return 1

    if arguments:
        print(classify(" ".join(arguments), model, vectorizer))
        return 0

    run_interactive(model, vectorizer, read=read)
    return 0


if __name__ == "__main__":
    sys.exit(main())
