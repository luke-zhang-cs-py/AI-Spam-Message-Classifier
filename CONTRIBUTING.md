# Contributing

## Setup

```bash
pip install -r requirements.txt
python app.py            # http://127.0.0.1:5002
```

## The one thing that will confuse you

`spam_model.joblib` and `vectorizer.joblib` are **not in the repository** —
they are gitignored build output. `dataset.csv` is committed precisely so
that they can be rebuilt, and `ensure_model()` trains them on first use.

So a fresh clone has no model, and the first request or first test that needs
one pays about ten seconds to train it. That is the state CI is always in.

It is worth knowing before you write a test: asserting that
`spam_model.joblib` exists passes on your machine, where an earlier run left
one behind, and fails in every fresh clone. `test_artifacts_are_found_from_any_directory`
did exactly that. If a test needs a model, call `ensure_model()` and let it
build one.

```bash
git clone . /tmp/fresh && cd /tmp/fresh && pytest -q   # what CI sees
```

## Tests

```bash
pytest -q
python -m flake8 . --select=E9,F63,F7,F82
```

47 tests. The model is trained once per session rather than per test, so the
suite runs in seconds instead of minutes.

## Conventions

Every test that names a bug describes one that was really in this
repository — that is deliberate, and it is worth keeping. Prefer a test whose
name states the defect over one named after the function it calls.
