"""
train_spam_classifier.py
-------------------------
Train the filter from the corpus and save the artifacts.

    python -m cli.train_spam_classifier
    python -m cli.train_spam_classifier --data data/            # a shard directory
    python -m cli.train_spam_classifier --data dataset.csv      # one file
    python -m cli.train_spam_classifier --embeddings            # + transformer

This used to carry its own copy of the pipeline: `clean_text`, `load_data`,
`train_and_evaluate`, `save_model`, `load_artifacts` and `predict_message`
were all defined here *and* in `spam_classifier_all_in_one.py`, and a test
asserted the copies matched. The implementation is now in `spamlib.py` and
this file is the command-line entry point to it.

The names are still importable from here, because `classify.py` imports
MODEL_PATH, VECTORIZER_PATH and predict_message from this module.
"""

import argparse
import sys

from pipeline import spamlib
from pipeline.spamlib import (DATA_DIR, EMBEDDING_C, LEGACY_DATA_PATH, MAX_ITER, MIN_DF,
                     MODEL_PATH, NGRAM_RANGE, PRECISION_FLOOR, RANDOM_STATE,
                     STOP_WORDS, TEST_SIZE, VECTORIZER_PATH, Thresholded,
                     build_vectorizer, clean_text, compare_models,
                     describe_features, frame_features, load_data, metrics,
                     predict_message, spam_probability, train_and_evaluate,
                     vectorize, wants_raw_text)


def load_artifacts(model_path=None, vectorizer_path=None):
    """The saved pair, resolved against *this module's* paths. See the twin
    of this function in spam_classifier_all_in_one for why it is a wrapper
    and not a re-export."""
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


# Kept for callers that read it from here. The corpus is a directory now, so
# this is the fallback path rather than the primary one.
DATA_PATH = LEGACY_DATA_PATH

__all__ = [
    "DATA_DIR", "DATA_PATH", "EMBEDDING_C", "LEGACY_DATA_PATH", "MAX_ITER",
    "MIN_DF",
    "MODEL_PATH", "NGRAM_RANGE", "PRECISION_FLOOR", "RANDOM_STATE",
    "STOP_WORDS", "TEST_SIZE",
    "VECTORIZER_PATH", "Thresholded", "build_vectorizer", "clean_text",
    "compare_models", "describe_features", "frame_features",
    "load_artifacts", "load_data", "metrics", "predict_message",
    "save_model", "spam_probability", "train_and_evaluate", "vectorize",
    "wants_raw_text", "main",
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the spam filter and save model + vectorizer.")
    parser.add_argument("--data", default=None,
                        help="corpus directory or CSV (default: data/)")
    parser.add_argument("--quiet", action="store_true",
                        help="train without the comparison report")
    parser.add_argument("--embeddings", action="store_true",
                        help="add sentence-transformer features (needs the "
                             "optional extras and a local encoder)")
    args = parser.parse_args(argv)

    if args.embeddings:
        problem = spamlib.embedding_backend_problem()
        if problem:
            print("--embeddings is not available: %s" % problem,
                  file=sys.stderr)
            return 1

    try:
        df = load_data(args.data)
    except FileNotFoundError as problem:
        print(problem, file=sys.stderr)
        return 1

    if not args.quiet:
        print("Loaded %d messages: %s"
              % (len(df), dict(df["label"].value_counts())))
        print()

    model, vectorizer = train_and_evaluate(df, verbose=not args.quiet,
                                           embeddings=args.embeddings)
    save_model(model, vectorizer)
    print("Saved %s" % MODEL_PATH)
    print("Saved %s" % VECTORIZER_PATH)
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
