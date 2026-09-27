"""Is the optional embedding backend worth its two gigabytes?

    python tools/compare_backends.py
    python tools/compare_backends.py --sweep-c

This is the measurement the decision to keep `--embeddings` off by default
rests on, and the reason the numbers quoted in `embeddings.py` and `README.md`
can be re-derived rather than trusted. Run it whenever the corpus grows: the
whole argument for keeping the backend is that the gap should move as the
data does, and that is a claim with an expiry date on it.

Everything is scored the way `spamlib` scores: one 5-fold stratified split
shared by every row of the table, the features and the estimator refitted
inside each fold so no message is judged by a fold that learned anything
from it -- vocabulary and idf included -- and the threshold chosen
to maximise F1 subject to precision >= PRECISION_FLOOR. Comparing a tuned
model against an untuned one at 0.5 is how a backend gets adopted on the
strength of its threshold.

`--sweep-c` answers the narrower question of where EMBEDDING_C should sit.
A constant with a round number and no measurement behind it is a magic
number, and this is the file that stops it being one.

Not shipped code: this is in tools/, which .coveragerc omits.
"""
import argparse
import os
import sys
import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import embeddings as backend          # noqa: E402
from pipeline import spamlib                        # noqa: E402

# Only for the "embeddings alone" row. Nothing spamlib builds is purely
# dense, so this is the one estimator in the file that the library itself
# never constructs -- kept because the row it produces is the honest half of
# the comparison and the one that makes the backend look bad.
from sklearn.svm import SVC           # noqa: E402


def scored(estimator, folds, y):
    """(threshold, precision, recall, f1), out of fold, at the floor."""
    scores = spamlib._out_of_fold_scores(estimator, folds, y)
    return spamlib.choose_threshold(scores, y)


def features(df, y, lexical, semantic):
    """Per-fold features with either half, or both, the extractor refitted
    on every training fold exactly as `spamlib.compare_models` does."""
    tfidf = spamlib.build_vectorizer() if lexical else None
    if not semantic:
        return spamlib.fold_features(tfidf, df, y)
    combined = backend.SemanticFeatures(lexical=tfidf,
                                        cleaner=spamlib.clean_text)
    return spamlib.fold_features(combined, df, y)


def row(label, name, estimator, folds, y):
    cut, precision, recall, f1 = scored(estimator, folds, y)
    print("  %-20s %-30s %9.4f %10.3f %8.3f %8.3f"
          % (label, name, cut, precision, recall, f1))
    return f1, precision, recall, label, name


def compare(df, y):
    print("=" * 94)
    print("  %-20s %-30s %9s %10s %8s %8s"
          % ("features", "estimator", "threshold", "precision", "recall", "F1"))
    print("  " + "-" * 90)

    rows = []
    lexical = features(df, y, lexical=True, semantic=False)
    rows.append(row("tf-idf", "MultinomialNB", MultinomialNB(), lexical, y))
    rows.append(row("tf-idf", "ComplementNB", ComplementNB(), lexical, y))
    rows.append(row("tf-idf", "LogReg",
                    LogisticRegression(max_iter=spamlib.MAX_ITER), lexical, y))

    dense = features(df, y, lexical=False, semantic=True)
    C = spamlib.EMBEDDING_C
    rows.append(row("embeddings", "LogReg",
                    LogisticRegression(max_iter=spamlib.MAX_ITER, C=C),
                    dense, y))
    rows.append(row("embeddings", "RBF SVM",
                    SVC(kernel="rbf", C=C, probability=True,
                        random_state=spamlib.RANDOM_STATE), dense, y))

    both = features(df, y, lexical=True, semantic=True)
    for name, estimator in spamlib.candidate_models(embeddings=True).items():
        rows.append(row("tf-idf + embeddings", name, estimator, both, y))

    rows.sort(reverse=True)
    print()
    print("=" * 94)
    best = rows[0]
    incumbent = [r for r in rows if r[3] == "tf-idf"
                 and r[4] == "MultinomialNB"][0]
    print("best      : %-19s %-30s F1 %.3f  precision %.3f  recall %.3f"
          % (best[3], best[4], best[0], best[1], best[2]))
    print("default   : %-19s %-30s F1 %.3f  precision %.3f  recall %.3f"
          % (incumbent[3], incumbent[4], incumbent[0], incumbent[1],
             incumbent[2]))
    print("difference: %+.4f F1" % (best[0] - incumbent[0]))
    print()
    # This used to print a fixed verdict -- "about three thousandths of F1,
    # which is not a result" -- and went on printing it after refitting the
    # vectoriser per fold moved the margin to +0.026. A tool whose job is to
    # re-check a claim should not have the old claim typed into it, so the
    # verdict is left to whoever reads the two rows above.
    print("Near F1 0.85 on %d messages, one spam changing side moves F1 by"
          " about 0.003." % len(df))


def sweep_c(df, y):
    """Where EMBEDDING_C should sit, on the features it is actually used on."""
    both = features(df, y, lexical=True, semantic=True)
    print()
    print("=" * 94)
    print("EMBEDDING_C, balanced logistic regression on tf-idf + embeddings")
    print("  %10s %10s %8s %8s" % ("C", "precision", "recall", "F1"))
    print("  " + "-" * 40)
    for candidate in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
        estimator = LogisticRegression(max_iter=spamlib.MAX_ITER,
                                       C=candidate, class_weight="balanced")
        _cut, precision, recall, f1 = scored(estimator, both, y)
        marker = "  <- EMBEDDING_C" if candidate == spamlib.EMBEDDING_C else ""
        print("  %10.1f %10.3f %8.3f %8.3f%s"
              % (candidate, precision, recall, f1, marker))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-c", action="store_true",
                        help="also sweep the regularisation constant")
    args = parser.parse_args(argv)

    problem = spamlib.embedding_backend_problem()
    if problem:
        print("cannot compare: %s" % problem, file=sys.stderr)
        return 1

    df = spamlib.load_data()
    y = df["label"].map(spamlib.LABELS).to_numpy()
    print("%d messages, %s" % (len(df), dict(df["label"].value_counts())))
    print("floor: precision >= %.2f, %d-fold, seed %d"
          % (spamlib.PRECISION_FLOOR, spamlib.CV_FOLDS, spamlib.RANDOM_STATE))
    print()

    started = time.time()
    compare(df, y)
    if args.sweep_c:
        sweep_c(df, y)
    print()
    print("measured in %.0fs" % (time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
