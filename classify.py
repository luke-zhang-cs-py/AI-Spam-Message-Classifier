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
    confidence)" of its own invention while the shared version returned a
    differently formatted string. One decision, three entry points, three
    spellings of the answer.

    The logic is imported now, so this file only supplies the
    capitalisation. `predict_message` itself no longer reports a confidence
    figure at all -- it returns a bare "spam" or "ham" -- so `verdict` has no
    text after the label, `rest` is always empty, and the actual output here
    is just "SPAM" or "HAM".
    """
    verdict = predict_message(message, model, vectorizer)
    label, _, rest = verdict.partition(" ")
    return f"{label.upper()} {rest}".strip()


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
        print("Model not found. Run `python train_spam_classifier.py` first.")
        return 1

    if arguments:
        print(classify(" ".join(arguments), model, vectorizer))
        return 0

    run_interactive(model, vectorizer, read=read)
    return 0


if __name__ == "__main__":
    sys.exit(main())
