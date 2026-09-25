"""The command-line halves of the command-line tools, run in process.

There were already tests for these, and they drove the scripts through
`subprocess.run` -- which proves the real thing works from a real shell and
is worth keeping, but contributes nothing to coverage, because the lines
execute in a process the measurement never sees. The result was that
`spam_classifier_all_in_one.py` sat at 38% and `train_spam_classifier.py` at
59% while both were, in practice, exercised end to end on every run.

So these call `main()`, `run_training_and_demo()` and `run_interactive()`
directly. Two consequences worth stating:

**The artifact paths are redirected in every test that trains.** The module
constants are monkeypatched, which is the entire reason `save_model` and
`load_artifacts` are thin wrappers in both façades rather than re-exports --
a re-exported function closes over `spamlib`'s constants and ignores the
patch, so the artifacts would land in the project directory and the next
test would score against them.

**The interactive loops take their reader as a parameter.** Both used to
call `input()` directly. `spam_classifier_all_in_one` was fixed for that a
while ago; `classify.py` was not, and its loop was the largest untestable
block in the project -- the whole interactive half of an interactive tool,
runnable only by hand.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cli import classify as classify_cli                 # noqa: E402
import spam_classifier_all_in_one as allinone   # noqa: E402
from pipeline import spamlib                                  # noqa: E402
from cli import train_spam_classifier as trainer         # noqa: E402


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Redirect every module's artifact paths at a temporary directory.

    All four modules, not just the one under test: `classify` reads the
    paths it imported from `train_spam_classifier`, and a test that patches
    one and not the others has half the project writing into the real
    project directory.
    """
    model = str(tmp_path / "m.joblib")
    vector = str(tmp_path / "v.joblib")
    for module in (spamlib, allinone, trainer, classify_cli):
        monkeypatch.setattr(module, "MODEL_PATH", model, raising=False)
        monkeypatch.setattr(module, "VECTORIZER_PATH", vector, raising=False)
    return model, vector


class Typed:
    """A stand-in for `input`, and a record of what it was asked.

    Raises whatever it is given as a final value, so the EOF and Ctrl-C
    paths are reachable: those are how the loops actually end, and a fake
    that only ever returns strings leaves the exit route uncovered.
    """

    def __init__(self, *lines, ending=EOFError):
        self.lines = list(lines)
        self.ending = ending
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.lines:
            raise self.ending
        return self.lines.pop(0)


# --------------------------------------------------------------- classify.py

def test_the_classify_cli_labels_a_message_from_its_arguments(artifacts):
    """The argv path. Covered by a subprocess test as well, which proves it
    works from a shell; this one is what measures it."""
    trainer.main(["--quiet"])
    assert classify_cli.main(["WINNER! Claim your FREE prize now"]) == 0


def test_the_classify_cli_joins_its_arguments(artifacts, capsys):
    """An unquoted message arrives as several arguments, which is the
    ordinary way to call this by mistake and should still work.

    The words are chosen so the test can fail. The first version used
    ["Claim", "your", "FREE", "prize", "now"], and replacing
    `" ".join(arguments)` with `arguments[0]` left it green -- "Claim" on
    its own already classifies as spam, so the assertion held whether the
    arguments were joined or not. These words only reach a spam verdict
    together, and the first assertion is the negative control that says so.
    """
    trainer.main(["--quiet"])
    words = ["You", "have", "won", "a", "free", "cruise", "click", "here"]

    model, vectorizer = classify_cli.load_artifacts()
    assert classify_cli.classify(words[0], model, vectorizer).upper().startswith("HAM"), (
        "the control failed: the first word alone is already spam, so this "
        "test could pass without joining anything")

    capsys.readouterr()
    assert classify_cli.main(words) == 0
    assert capsys.readouterr().out.strip().upper().startswith("SPAM")


def test_the_classify_prompt_classifies_each_line(artifacts, capsys):
    """The interactive loop, which was unreachable from a test until its
    reader became a parameter."""
    trainer.main(["--quiet"])
    capsys.readouterr()

    typed = Typed("WINNER! Claim your FREE prize now",
                  "   ",                        # whitespace: skipped, no crash
                  "are we still on for dinner")
    assert classify_cli.main([], read=typed) == 0

    printed = capsys.readouterr().out
    assert typed.prompts == ["> "] * 4, "the prompt stopped being shown"
    assert printed.count("SPAM") + printed.count("HAM") == 2, (
        "the blank line was classified, or a real one was not")
    assert "Bye!" in printed


def test_the_classify_prompt_ends_on_ctrl_c_as_well_as_eof(artifacts, capsys):
    """Both endings, because the loop catches both and a test of one leaves
    the other branch of the same `except` untested."""
    trainer.main(["--quiet"])
    model, vectorizer = classify_cli.load_artifacts()
    for ending in (EOFError, KeyboardInterrupt):
        capsys.readouterr()
        classify_cli.run_interactive(model, vectorizer, read=Typed(ending=ending))
        assert "Bye!" in capsys.readouterr().out, ending


def test_the_classify_cli_reports_a_missing_model_with_a_nonzero_status(
        artifacts, capsys):
    """The state of every fresh clone: the .joblib files are gitignored."""
    assert classify_cli.main(["anything"]) == 1
    assert "cli.train_spam_classifier" in capsys.readouterr().out, (
        "the message does not say what to run")


# ------------------------------------------------- train_spam_classifier.py

def test_the_trainer_main_trains_and_saves(artifacts, capsys):
    model_path, vector_path = artifacts
    assert trainer.main([]) == 0
    assert os.path.isfile(model_path) and os.path.isfile(vector_path)

    printed = capsys.readouterr().out
    assert "Loaded 412 messages" in printed or "Loaded" in printed
    assert "MODEL COMPARISON" in printed, "the report was not printed"
    assert "Saved" in printed


def test_the_trainer_quiet_mode_still_saves_but_says_less(artifacts, capsys):
    model_path, _ = artifacts
    assert trainer.main(["--quiet"]) == 0
    assert os.path.isfile(model_path)

    printed = capsys.readouterr().out
    assert "MODEL COMPARISON" not in printed
    assert "Saved" in printed, "--quiet should still confirm the write"


def test_the_trainer_accepts_one_csv_instead_of_the_shard_directory(
        artifacts, tmp_path, capsys):
    """Five of each label, not two: model choice is `CV_FOLDS`-fold and
    stratified, so a class with fewer members than there are folds makes
    StratifiedKFold raise before anything gets trained. A four-row fixture
    is the smallest thing that looks reasonable and the smallest thing that
    cannot work."""
    rows = ["label,text"]
    for index in range(spamlib.CV_FOLDS):
        rows.append("spam,win a free prize now number %d" % index)
        rows.append("ham,see you at six on %d" % index)
    (tmp_path / "tiny.csv").write_text("\n".join(rows) + "\n",
                                       encoding="utf-8")
    assert trainer.main(["--data", str(tmp_path / "tiny.csv"), "--quiet"]) == 0


def test_the_trainer_reports_a_missing_corpus_rather_than_a_traceback(
        artifacts, tmp_path, monkeypatch, capsys):
    """Both the shard directory and the legacy CSV absent, which is what a
    checkout with no data/ and no dataset.csv looks like."""
    monkeypatch.setattr(spamlib, "DATA_DIR", str(tmp_path / "no-shards"))
    monkeypatch.setattr(spamlib, "LEGACY_DATA_PATH",
                        str(tmp_path / "no-dataset.csv"))
    assert trainer.main([]) == 1
    assert "no corpus" in capsys.readouterr().err


def test_the_trainers_load_artifacts_wrapper_reads_its_own_paths(artifacts):
    """The wrapper exists so a monkeypatched MODEL_PATH is honoured. It was
    at 0% -- nothing had ever called this copy of it, so the property it
    exists for was unverified."""
    assert trainer.load_artifacts() == (None, None), "nothing saved yet"
    trainer.main(["--quiet"])
    model, vectorizer = trainer.load_artifacts()
    assert model is not None and vectorizer is not None
    assert spamlib.predict_message("claim your free prize", model,
                                   vectorizer) in ("spam", "ham")


# --------------------------------------------- spam_classifier_all_in_one.py

def test_the_all_in_one_trains_reports_and_demos(artifacts, capsys):
    model_path, vector_path = artifacts
    model, vectorizer = allinone.run_training_and_demo()
    assert os.path.isfile(model_path) and os.path.isfile(vector_path)

    printed = capsys.readouterr().out
    assert "Loading corpus" in printed
    assert "MODEL COMPARISON" in printed
    assert "DEMO" in printed
    # Every demo message gets a line, and the adversarial ones are the point
    # of the demo: a bare list of six obvious messages tells a reader
    # nothing they could not have guessed.
    for message in allinone.DEMO_MESSAGES:
        assert message[:40] in printed, message


def test_the_all_in_one_can_train_without_saying_anything(artifacts, capsys):
    allinone.run_training_and_demo(verbose=False)
    assert capsys.readouterr().out == "", "verbose=False still printed"


def test_the_all_in_one_prompt_shows_the_probability_and_the_cut(artifacts,
                                                                 capsys):
    """This loop's rendering is deliberately not the one in classify.py: it
    shows the probability and the threshold, which is what makes a verdict
    near the cut legible."""
    allinone.run_training_and_demo(verbose=False)
    model, vectorizer = allinone.load_artifacts()

    capsys.readouterr()
    allinone.run_interactive(model, vectorizer,
                             read=Typed("claim your free prize now"))
    printed = capsys.readouterr().out
    assert "p=" in printed and "cut" in printed
    assert "SPAM" in printed or "HAM" in printed


def test_the_all_in_one_prompt_stops_on_an_empty_line(artifacts, capsys):
    """An empty line ends this loop, where classify.py skips it. Different
    contracts, both intended, both now covered."""
    allinone.run_training_and_demo(verbose=False)
    model, vectorizer = allinone.load_artifacts()

    typed = Typed("", "never read")
    allinone.run_interactive(model, vectorizer, read=typed)
    assert typed.lines == ["never read"], "it kept reading past the blank line"


def test_the_all_in_one_prompt_ends_on_ctrl_c_as_well_as_eof(artifacts):
    allinone.run_training_and_demo(verbose=False)
    model, vectorizer = allinone.load_artifacts()
    for ending in (EOFError, KeyboardInterrupt):
        allinone.run_interactive(model, vectorizer, read=Typed(ending=ending))


def test_the_all_in_one_main_trains_then_demos(artifacts):
    assert allinone.main([]) == 0


def test_the_all_in_one_main_can_go_straight_to_the_prompt(
        artifacts, monkeypatch):
    """--interactive trains first and then reads.

    The reader is threaded through `main` rather than installed over
    `builtins.input`. Patching the builtin was the first attempt and it did
    nothing: both loops used to declare `read=input`, and a default argument
    is evaluated once at import, so the signature had already captured the
    real builtin and the patch was invisible to it. The loops resolve
    `input` at call time now, so either approach works -- but passing it is
    the one that says what the test means.
    """
    typed = Typed("claim your free prize", ending=EOFError)
    assert allinone.main(["--interactive"], read=typed) == 0
    assert typed.prompts, "the prompt was never shown"

    # And the builtin route works too, now that it is resolved at call time.
    monkeypatch.setattr("builtins.input", Typed(ending=EOFError))
    assert allinone.main(["--interactive"]) == 0


def test_the_all_in_one_classifies_one_message_from_the_saved_model(
        artifacts, capsys):
    allinone.run_training_and_demo(verbose=False)
    capsys.readouterr()
    assert allinone.main(["--classify", "WINNER claim your free prize"]) == 0
    assert capsys.readouterr().out.strip() in ("spam", "ham")


def test_message_is_accepted_as_a_spelling_of_classify(artifacts, capsys):
    """--classify is the documented name; --message is the obvious guess,
    and the parser accepts both into the same destination."""
    allinone.run_training_and_demo(verbose=False)
    capsys.readouterr()
    assert allinone.main(["--message", "see you at six"]) == 0
    assert capsys.readouterr().out.strip() in ("spam", "ham")


def test_classifying_without_a_saved_model_says_what_to_run(
        artifacts, capsys):
    assert allinone.main(["--classify", "anything"]) == 1
    assert "without --classify" in capsys.readouterr().err


def test_the_source_of_both_mains_is_reachable_without_a_subprocess():
    """A guard on this file's premise.

    The point of it is that these entry points are measured rather than only
    proven to work in a subprocess. If someone puts the argument parsing
    back behind `sys.argv` with no parameter, every test above still passes
    -- by driving a default that no longer means anything -- so the
    signatures are asserted.
    """
    import inspect
    assert "argv" in inspect.signature(allinone.main).parameters
    assert "argv" in inspect.signature(trainer.main).parameters
    assert "argv" in inspect.signature(classify_cli.main).parameters
    for function in (classify_cli.run_interactive, allinone.run_interactive):
        assert "read" in inspect.signature(function).parameters, function
