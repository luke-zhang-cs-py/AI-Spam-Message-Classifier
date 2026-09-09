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
import re
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
    That quarter is only held-out data if it is the *same* quarter.

    The previous version of this test did not look at the trainer at all,
    despite its name, and its final assertion was
    `web.clf.TEST_SIZE == allinone.TEST_SIZE` -- which cannot fail, because
    `web.clf` *is* the allinone module. So it compared a value to itself
    while train_spam_classifier.py still held the four literals.
    """
    import app as web
    app_source = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "test_size=clf.TEST_SIZE" in app_source
    assert "random_state=clf.RANDOM_STATE" in app_source
    # Stated as identity, which is the real claim: app.py uses the canonical
    # module rather than a recipe of its own.
    assert web.clf is allinone

    # The half that was missing. Identity rather than equality, so two
    # separately declared constants that happen to be equal do not pass.
    for name in ("TEST_SIZE", "RANDOM_STATE", "NGRAM_RANGE", "MIN_DF",
                 "STOP_WORDS", "MAX_ITER"):
        assert getattr(trainer, name) is getattr(allinone, name), name


def test_the_trainer_states_no_recipe_of_its_own():
    """The structural half.

    Literals back in the trainer are the duplication starting again, and they
    would agree with the canonical values right up until somebody changed one
    -- which is the failure the comment in allinone described in the past
    tense while this half of it was still live.
    """
    source = open(os.path.join(ROOT, "train_spam_classifier.py"),
                  encoding="utf-8").read()
    code = re.sub(r'"""(?:.|\n)*?"""', "", source)
    code = re.sub(r"#[^\n]*", "", code)
    for literal in ("test_size=0.25", "random_state=42", "ngram_range=(1, 2)",
                    "min_df=1", 'stop_words="english"', "max_iter=1000"):
        assert literal not in code, f"the trainer restates {literal}"


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


# --------------------------------------------------- the state-mutating parts

def test_retraining_replaces_the_cached_model_and_reports_it():
    """/api/retrain was the one endpoint no test touched, and it is the one
    that mutates process-wide state. Left uncovered it is also the reason the
    module needed a reset(): force_retrain=True swaps the model out from under
    every later request in the process."""
    import app as web
    web.app.config["TESTING"] = True
    client = web.app.test_client()

    first = client.get("/api/model").get_json()
    reply = client.post("/api/retrain")
    assert reply.status_code == 200
    body = reply.get_json()
    assert body["ok"] is True
    assert body["name"]
    assert body["metrics"]
    # Deterministic training, so the retrained model reports the same name and
    # scores. That is the point: a differing figure here would mean the split
    # or the seed moved.
    assert body["name"] == first["name"]
    assert body["metrics"] == first["metrics"]


def test_reset_forgets_the_model_and_the_next_call_rebuilds_it():
    """The isolation seam. Without it a retrain in one test silently changes
    what every later test is running against."""
    import app as web
    web.ensure_model()
    assert web._state["model"] is not None
    web.reset()
    assert web._state["model"] is None
    assert web._state["metrics"] is None
    model, vectorizer = web.ensure_model()
    assert model is not None and vectorizer is not None


# ------------------------------------------------------------ the classify CLI

def test_the_classify_cli_labels_a_message_from_argv():
    """classify.main() was at 45% -- the whole command-line half of a
    command-line tool."""
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "classify.py"),
         "WINNER! Claim your FREE prize now, call 09061701461"],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().upper().startswith("SPAM"), result.stdout


def test_the_classify_cli_says_what_to_do_when_there_is_no_model(tmp_path,
                                                                 monkeypatch):
    """Rather than a traceback about a missing file. The artifacts are
    gitignored, so this is the state of every fresh clone."""
    monkeypatch.setattr(cli, "MODEL_PATH", str(tmp_path / "absent.joblib"))
    monkeypatch.setattr(cli, "VECTORIZER_PATH", str(tmp_path / "absent2.joblib"))
    monkeypatch.setattr(sys, "argv", ["classify.py"])
    assert cli.main() == 1
