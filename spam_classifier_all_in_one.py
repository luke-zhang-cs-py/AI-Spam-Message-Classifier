"""
spam_classifier_all_in_one.py
------------------------------
The end-to-end demo: train from the corpus, report, and classify.

    python spam_classifier_all_in_one.py            # train, then a short demo
    python spam_classifier_all_in_one.py --interactive
    python spam_classifier_all_in_one.py --message "you have won a prize"
    python spam_classifier_all_in_one.py --embeddings   # + the transformer

What it no longer is
====================
This file used to be the name the project was organised around. It held the
pipeline -- `clean_text`, `load_data`, `train_and_evaluate`, `save_model`,
`load_artifacts`, `predict_message` -- and an 81-message dataset embedded as
a string literal. `train_spam_classifier.py` held its own copy of all six,
`app.py` imported from here, `classify.py` imported from there, and a test
asserted the two copies agreed.

That test kept the copies in step without ever reducing how many there were,
so a change to the pipeline meant three edits and a test to confirm you had
made them all. The implementation now lives once, in `spamlib.py`, and the
embedded dataset is gone: the corpus is `data/*.csv`, which is where new
material goes.

The names below are re-exported rather than removed, because `app.py` and
the tests import them from this module and a refactor should not have to be
a rename as well.
"""

import argparse
import sys

import spamlib
# Re-exported for the callers that already import them from here. Spelled
# out rather than star-imported so what this module promises is readable.
from spamlib import (EMBEDDING_C, MAX_ITER, MIN_DF, MODEL_PATH, NGRAM_RANGE,
                     PRECISION_FLOOR, RANDOM_STATE, STOP_WORDS, TEST_SIZE,
                     VECTORIZER_PATH, Thresholded, build_vectorizer,
                     clean_text, compare_models, describe_features,
                     frame_features, load_data, metrics, predict_message,
                     spam_probability, train_and_evaluate, vectorize,
                     wants_raw_text)


def load_artifacts(model_path=None, vectorizer_path=None):
    """The saved pair, resolved against *this module's* paths.

    A thin wrapper rather than a re-export, because the tests monkeypatch
    MODEL_PATH on this module to check the missing-file behaviour. A
    re-exported function closes over spamlib's own constants and ignores
    that, so the patch would silently do nothing and the test would pass
    against the real model on disk.
    """
    return spamlib.load_artifacts(model_path or MODEL_PATH,
                                  vectorizer_path or VECTORIZER_PATH)


def save_model(model, vectorizer, model_path=None, vectorizer_path=None):
    """Write the pair to *this module's* paths.

    A wrapper for the same reason load_artifacts is one: the tests
    monkeypatch MODEL_PATH on this module, and a re-exported function reads
    spamlib's constants instead, so the artifacts would land in the real
    project directory while the test looked in its tmp_path.
    """
    return spamlib.save_model(model, vectorizer,
                              model_path or MODEL_PATH,
                              vectorizer_path or VECTORIZER_PATH)


__all__ = [
    "EMBEDDING_C", "MAX_ITER", "MIN_DF", "MODEL_PATH", "NGRAM_RANGE",
    "PRECISION_FLOOR", "RANDOM_STATE",
    "STOP_WORDS", "TEST_SIZE", "VECTORIZER_PATH", "Thresholded",
    "build_vectorizer", "clean_text",
    "compare_models", "describe_features", "frame_features",
    "load_artifacts", "load_data", "metrics",
    "predict_message", "save_model", "spam_probability", "train_and_evaluate",
    "vectorize", "wants_raw_text",
    "run_training_and_demo", "run_interactive", "main",
]

# Half of these are adversarial on purpose. A demo of six obvious messages
# tells a reader nothing they could not have guessed; "I won the pub quiz"
# and a legitimate passcode text are the ones worth watching, because they
# are the ones a bag-of-words filter gets wrong.
DEMO_MESSAGES = [
    "Congratulations! You've won a free cruise. Click here to claim now",
    "Hey, are we still on for dinner at 7?",
    "URGENT: your account will be suspended. Verify your details immediately",
    "I won the pub quiz on my own, absolutely buzzing",
    "Your parcel is held pending a 2.99 customs fee. Pay at rm-fees.co",
    "Your one time passcode is 481902. It expires in 10 minutes",
]


def run_training_and_demo(verbose=True, embeddings=False):
    """Train from the corpus, save the artifacts, classify the demo set."""
    if verbose:
        print("Loading corpus...")
    df = load_data()
    if verbose:
        shards = len(spamlib.data_files()) or 1
        print("  %d messages from %d file(s): %s"
              % (len(df), shards, dict(df["label"].value_counts())))
        print()

    model, vectorizer = train_and_evaluate(df, verbose=verbose,
                                           embeddings=embeddings)
    save_model(model, vectorizer)
    if verbose:
        print("Saved %s and %s" % (MODEL_PATH, VECTORIZER_PATH))
        print()
        print("=" * 72)
        print("DEMO")
        print("=" * 72)
        for message in DEMO_MESSAGES:
            print("  %-5s  p=%.3f  %s"
                  % (predict_message(message, model, vectorizer),
                     spam_probability(message, model, vectorizer),
                     message[:56]))
        print()
    return model, vectorizer


def run_interactive(model, vectorizer, read=input):
    """Classify messages typed at a prompt until EOF or an empty line.

    `read` is a parameter so this is testable without a terminal. It used to
    call input() directly, which made the loop unreachable from a test and
    therefore unverified.
    """
    print("Type a message and press enter. Empty line or Ctrl-C to quit.")
    while True:
        try:
            message = read("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            return
        print("  %s  (p=%.3f, cut %.4f)"
              % (predict_message(message, model, vectorizer).upper(),
                 spam_probability(message, model, vectorizer),
                 getattr(model, "threshold", 0.5)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and demo the filter.")
    parser.add_argument("--interactive", action="store_true",
                        help="classify messages typed at a prompt")
    # --classify is the documented name and what the tests drive; --message
    # is accepted as well because it is the obvious guess.
    parser.add_argument("--classify", "--message", dest="classify",
                        help="classify one message with the saved model")
    # Off by default because it was measured: +0.003 F1 for ~2 GB of
    # dependencies, and worse than tf-idf on its own. See embeddings.py.
    parser.add_argument("--embeddings", action="store_true",
                        help="add sentence-transformer features (needs the "
                             "optional extras and a local encoder)")
    args = parser.parse_args(argv)

    if args.classify:
        model, vectorizer = load_artifacts()
        if model is None:
            print("No saved model. Run this without --classify first.",
                  file=sys.stderr)
            return 1
        # Deliberately no --embeddings here. Which backend to use is a
        # property of the artifact on disk, not of the command that reads
        # it: `vectorize` asks the loaded vectorizer what it wants, so
        # classifying an embedding-backed model needs no flag and passing
        # one could only contradict the file.
        print(predict_message(args.classify, model, vectorizer))
        return 0

    if args.embeddings:
        problem = spamlib.embedding_backend_problem()
        if problem:
            print("--embeddings is not available: %s" % problem,
                  file=sys.stderr)
            return 1

    model, vectorizer = run_training_and_demo(embeddings=args.embeddings)
    if args.interactive:
        run_interactive(model, vectorizer)
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
