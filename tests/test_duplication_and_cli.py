"""The two copies of the model code, and the two command-line entry points.

`spam_classifier_all_in_one.py` and `train_spam_classifier.py` define six of
the same functions. Three are byte-identical (`clean_text`,
`predict_message`, `save_model`), one has already drifted apart
cosmetically (`train_and_evaluate`), and two differ for real reasons (one
reads a CSV path, the other an embedded string; the CLIs are different).

Deduplicating them would defeat the point of the all-in-one file, whose
docstring opens "A single self-contained file that combines everything from
the project". So it stays -- and these tests are the price of that decision.
A copy nobody checks is how a fix lands in one file and not the other, which
is a mistake this project's history already contains.

`classify.py` and `train_spam_classifier.py` were both at 0% coverage.
"""

import os
import runpy
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import classify as cli                          # noqa: E402
import spam_classifier_all_in_one as allinone   # noqa: E402
import train_spam_classifier as trainer         # noqa: E402

MESSAGES = [
    "WINNER!! Claim your FREE prize now, call 09061701461",
    "Hey, are we still on for coffee at 6?",
    "URGENT! Your account has been suspended. Click http://bit.ly/x to verify",
    "can you send me the notes from class",
    "",
    "   ",
    "Text STOP to 88888 to unsubscribe. Msg&data rates apply.",
    "MiXeD CaSe, punctuation!!! and 12345 numbers",
    "ünïcödé and émojis 🎉 in a message",
]


# ------------------------------------------------- the duplication contract

def test_the_two_copies_of_clean_text_agree():
    """Byte-identical today. This is what notices if one of them changes."""
    for message in MESSAGES:
        assert allinone.clean_text(message) == trainer.clean_text(message), message


def test_the_two_copies_point_at_the_same_artifacts():
    """They write and read the same two files. A path fixed in one and not
    the other means one of them silently trains into the void."""
    assert allinone.MODEL_PATH == trainer.MODEL_PATH
    assert allinone.VECTORIZER_PATH == trainer.VECTORIZER_PATH


def test_both_artifact_paths_are_absolute():
    """They were bare relative names, so where the model landed depended on
    which directory you happened to be standing in."""
    for path in (allinone.MODEL_PATH, allinone.VECTORIZER_PATH,
                 trainer.MODEL_PATH, trainer.VECTORIZER_PATH, trainer.DATA_PATH):
        assert os.path.isabs(path), path


def test_the_two_copies_of_predict_message_agree():
    model, vectorizer = allinone.load_artifacts()
    assert model is not None, "run the trainer first"
    for message in MESSAGES:
        if not message.strip():
            continue
        assert (allinone.predict_message(message, model, vectorizer)
                == trainer.predict_message(message, model, vectorizer)), message


def test_the_web_app_cleans_text_the_same_way_as_the_cli():
    """Three entry points, one preprocessing step. If they disagree, the same
    message gets two different verdicts depending on how you asked."""
    import app as web
    for message in MESSAGES:
        body = web.clf.clean_text(message)
        assert body == allinone.clean_text(message) == trainer.clean_text(message)


def test_the_split_is_defined_once():
    """app.evaluate() rebuilds the trainer's split to score a loaded model.
    That quarter is only held-out data if it is the *same* quarter -- both
    numbers used to be written out separately in both files."""
    import app as web
    source = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "test_size=clf.TEST_SIZE" in source
    assert "random_state=clf.RANDOM_STATE" in source
    assert web.clf.TEST_SIZE == allinone.TEST_SIZE


# ----------------------------------------------------------------- classify

def test_classify_loads_the_saved_artifacts():
    model, vectorizer = cli.load_artifacts()
    assert model is not None and vectorizer is not None


def test_classify_labels_a_message():
    model, vectorizer = cli.load_artifacts()
    verdict = cli.classify("WINNER! Claim your FREE prize now", model, vectorizer)
    assert verdict.upper().startswith("SPAM"), verdict


def test_the_cli_renders_the_shared_verdict_rather_than_its_own():
    """It used to reimplement predict_message and return a differently
    formatted answer. Same decision, three entry points, three spellings."""
    model, vectorizer = cli.load_artifacts()
    message = "WINNER! Claim your FREE prize now"
    shared = allinone.predict_message(message, model, vectorizer)
    assert cli.classify(message, model, vectorizer).lower() == shared.lower()


def test_both_load_artifacts_have_the_same_contract(monkeypatch, tmp_path):
    """One of them printed and called sys.exit(1). Two functions with one
    name and opposite contracts is how you import a process-killer by
    accident -- and it made the function impossible to test.

    Checked by behaviour rather than by reading the source, because the
    source now contains the words "sys.exit" in a comment explaining this.
    """
    missing = str(tmp_path / "nope.joblib")
    for module in (cli, allinone):
        monkeypatch.setattr(module, "MODEL_PATH", missing)
        monkeypatch.setattr(module, "VECTORIZER_PATH", missing)
        assert module.load_artifacts() == (None, None), module.__name__


def test_classify_reports_missing_artifacts_rather_than_raising(monkeypatch):
    """Running it before training is the obvious first mistake, and it should
    say so rather than throw a traceback about a missing file."""
    monkeypatch.setattr(cli, "MODEL_PATH", os.path.join(ROOT, "not-a-model.joblib"))
    model, vectorizer = cli.load_artifacts()
    assert model is None and vectorizer is None


# ------------------------------------------------------------ the CLI paths

def run_module(args):
    """Run the all-in-one script the way a person would."""
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "spam_classifier_all_in_one.py"), *args],
        capture_output=True, text=True, timeout=300, cwd=os.path.expanduser("~"))


@pytest.mark.parametrize("message,expected", [
    ("WINNER! Claim your FREE prize now, call 09061701461", "spam"),
    ("Hey, are we still on for coffee at 6?", "ham"),
])
def test_the_all_in_one_cli_classifies(message, expected):
    """Also proves the absolute-path fix: cwd is the home directory, not the
    project, and it still finds its own model."""
    result = run_module(["--classify", message])
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().startswith(expected), result.stdout


def test_the_all_in_one_cli_rejects_nothing_to_do():
    """No arguments trains and demos; that is slow, so just check the parser
    accepts --help without exploding."""
    result = run_module(["--help"])
    assert result.returncode == 0
    assert "--classify" in result.stdout


def test_the_trainer_runs_end_to_end(tmp_path, monkeypatch):
    """train_spam_classifier.py was at 0% coverage: never once executed by
    anything but a person."""
    monkeypatch.setattr(trainer, "MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setattr(trainer, "VECTORIZER_PATH", str(tmp_path / "v.joblib"))

    df = trainer.load_data(trainer.DATA_PATH)
    assert len(df) > 0 and set(df["label"]) <= {"spam", "ham"}
    assert "clean_text" in df.columns

    model, vectorizer = trainer.train_and_evaluate(df)
    trainer.save_model(model, vectorizer)
    assert (tmp_path / "m.joblib").exists() and (tmp_path / "v.joblib").exists()

    verdict = trainer.predict_message("FREE entry! call now to claim", model, vectorizer)
    assert verdict.startswith(("spam", "ham"))


def test_the_trainer_drops_unlabelled_rows(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("label,text\nspam,buy now\n,orphan row\nham,hello\n", encoding="utf-8")
    df = trainer.load_data(str(csv))
    assert len(df) == 2, "the row with no label is dropped"


def test_the_embedded_dataset_matches_the_csv_columns():
    """The all-in-one carries its own copy of the data. Same shape, or the
    two entry points are training on different things."""
    embedded = allinone.load_data()
    external = trainer.load_data(trainer.DATA_PATH)
    assert list(embedded.columns) == list(external.columns)
    assert set(embedded["label"]) == set(external["label"]) == {"spam", "ham"}


def test_running_the_module_by_name_does_not_train_on_import():
    """Importing must not have side effects; the training is behind main()."""
    module = runpy.run_path(os.path.join(ROOT, "classify.py"), run_name="not_main")
    assert "main" in module
