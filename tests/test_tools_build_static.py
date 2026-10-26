"""tools/build_static.py, the build that publishes docs/app.

The build's whole claim is that it refuses to ship a JavaScript port that
disagrees with the Python. That claim was itself untested: the file was
omitted from coverage, so a check that had stopped being able to fail -- the
thing `--prove` exists to catch -- would only have been noticed by someone
running `--prove` by hand.

There is no browser here and no node, and CI has neither, so Playwright is
replaced. The stand-in is a *faithful port*: every `page.evaluate` is
answered with what the Python answered, which is what the real spam.js is
required to do. It also reads the page it was handed for the six
`--prove` sabotages and breaks its answers the way each injected line would,
so `prove()` can be shown catching all six -- and, with a sabotage the
stand-in is told to ignore, shown failing the build when one slips through.

What that does not test is the JavaScript. That is the build's job, run with
a real browser; these tests are about whether the build's comparisons,
refusals and output are what it says they are.

Every file the build writes goes to `tmp_path`: OUT is redirected, and so are
the artifact paths, so the committed docs/app and the project directory are
never touched.
"""
import contextlib
import copy
import gzip
import json
import os
import sys
import types

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import app as web                                         # noqa: E402
import build_static as build                              # noqa: E402
import spam_classifier_all_in_one as clf                  # noqa: E402
from pipeline import spamlib                              # noqa: E402

TEMPLATE = os.path.join(ROOT, "templates", "index.html")


# --------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def fitted():
    """(model, vectorizer): naive Bayes over the real corpus, at a cut the
    comparison corpus is guaranteed to straddle.

    Fitted directly rather than through `train_and_evaluate`, which picks
    whichever candidate wins on the installed scikit-learn. The build has to
    see a message between the cut and 0.5 or it refuses (pass 0), so the cut
    is placed at the highest score under 0.5 -- the same situation the real
    model is in, made certain rather than likely.
    """
    frame = spamlib.load_data()
    vectorizer = spamlib.build_vectorizer()
    X = spamlib.frame_features(vectorizer, frame, fit=True)
    y = frame["label"].map(spamlib.LABELS).to_numpy()
    estimator = MultinomialNB().fit(X, y)

    # One message at a time, through the same call python_reference makes,
    # so the score the cut is placed on is bit for bit the one it compares.
    unwrapped = spamlib.Thresholded(estimator)
    scores = [spamlib.spam_probability(m, unwrapped, vectorizer)
              for m in build.corpus()]
    cut = max(s for s in scores if s < 0.5)
    model = spamlib.Thresholded(estimator, cut, name="Multinomial Naive Bayes")
    return model, vectorizer


@pytest.fixture
def artifacts(fitted, tmp_path, monkeypatch):
    """The fitted pair saved to tmp_path, and every module pointed at it.

    spamlib's paths for `build.load_model`, the all-in-one's for the Flask
    app the build replays requests against. The app's cached model is
    dropped on the way in and out so no other test inherits this one.
    """
    model_path = str(tmp_path / "m.joblib")
    vector_path = str(tmp_path / "v.joblib")
    for module in (spamlib, clf):
        monkeypatch.setattr(module, "MODEL_PATH", model_path)
        monkeypatch.setattr(module, "VECTORIZER_PATH", vector_path)
    spamlib.save_model(*fitted, model_path=model_path,
                       vectorizer_path=vector_path)
    web.reset()
    yield model_path, vector_path
    web.reset()


def synthetic(threshold=0.45, probabilities=(0.2, 0.47, 0.9)):
    """A reference in the shape `python_reference` returns, small enough to
    reason about: one message under the cut, one in the band between the cut
    and 0.5, one well over."""
    messages = ["message %d" % index for index in range(len(probabilities))]
    payloads = []
    for message, probability in zip(messages, probabilities):
        spam = probability >= threshold
        payloads.append({
            "message": message, "cleaned": message,
            "label": "spam" if spam else "ham",
            "confidence": round(probability if spam else 1 - probability, 4),
            "knownTokens": 1,
            "tokens": [{"token": "message", "weight": 0.1, "tfidf": 1.0}],
        })
    return {
        "messages": messages,
        "cleaned": list(messages),
        "payloads": payloads,
        "verdicts": [p["label"] for p in payloads],
        "probabilities": list(probabilities),
        "threshold": threshold,
        "requests": [{"method": "GET", "url": "/api/model", "body": None},
                     {"method": "POST", "url": "/api/classify",
                      "body": {"message": messages[0]}}],
        "replies": [{"status": 200, "json": {"ok": True, "name": "NB"}},
                    {"status": 200, "json": {"ok": True, "label": "ham"}}],
    }


class Message:
    """A console message, as Playwright hands one to a listener."""

    def __init__(self, kind, text):
        self.type = kind
        self.text = text


class Page:
    """The slice of a Playwright page the build uses."""

    def __init__(self, browser):
        self.browser = browser
        self.handlers = {}
        self.html = None
        self.closed = False

    def on(self, event, handler):
        self.handlers[event] = handler

    def set_content(self, html):
        self.html = html
        # An ordinary console.log, which must not count as an error.
        self.handlers["console"](Message("log", "loaded"))
        for problem in self.browser.load_errors:
            self.handlers["pageerror"](RuntimeError(problem))

    def evaluate(self, script, arg=None):
        for problem in self.browser.check_errors:
            self.handlers["console"](Message("error", problem))
        return self.browser.answer(self, script, arg)

    def close(self):
        self.closed = True


class FaithfulPort:
    """A browser whose JavaScript gives exactly the Python's answers.

    `blind` names sabotages it does not reproduce -- a stand-in for a check
    with a hole in it. `load_errors` are thrown as the page loads, and
    `check_errors` are logged to the console during the checks.
    """

    def __init__(self, reference, blind=(), load_errors=(), check_errors=()):
        self.reference = reference
        self.blind = set(blind)
        self.load_errors = list(load_errors)
        self.check_errors = list(check_errors)
        self.pages = []
        self.closed = False

    def new_page(self):
        page = Page(self)
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True

    def answer(self, page, script, arg):
        reference = copy.deepcopy(self.reference)
        broken = {name for name, injection, _ in build.SABOTAGE
                  if injection in page.html and name not in self.blind}

        if script == build.THRESHOLD_JS:
            # applied-cut reports the honest cut; only `threshold` moves it.
            return reference["threshold"] + (1e-12 if "threshold" in broken
                                             else 0.0)
        if script == build.CLEAN_JS:
            assert arg == reference["messages"]
            cleaned = reference["cleaned"]
            if "shoutRatio" in broken:
                cleaned[0] = cleaned[0].upper()
            return cleaned
        if script == build.CLASSIFY_JS:
            assert arg == reference["messages"]
            out = []
            for payload, probability in zip(reference["payloads"],
                                            reference["probabilities"]):
                payload["probability"] = probability
                if "idf" in broken or "logprob" in broken:
                    payload["probability"] = probability + 1e-9
                if "applied-cut" in broken:
                    payload["label"] = "spam" if probability >= 0.5 else "ham"
                if "vocabulary" in broken and payload["tokens"]:
                    payload["tokens"][0]["token"] += "x"
                out.append(payload)
            return out
        if script == build.ROUTES_JS:
            assert arg == reference["requests"]
            return reference["replies"]
        raise AssertionError("the build evaluated a script nothing expects: "
                             + script[:60])


def install_playwright(monkeypatch, browser):
    """Put a stand-in `playwright.sync_api` where the build imports it.

    Returns the list of keyword arguments `chromium.launch` was called with.
    """
    launches = []

    class Chromium:
        def launch(self, **options):
            launches.append(options)
            return browser

    @contextlib.contextmanager
    def sync_playwright():
        yield types.SimpleNamespace(chromium=Chromium())

    package = types.ModuleType("playwright")
    api = types.ModuleType("playwright.sync_api")
    api.sync_playwright = sync_playwright
    package.sync_api = api
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)
    return launches


def stopped(call, *args, **kwargs):
    """The message a `build_static: ...` refusal carries."""
    with pytest.raises(SystemExit) as refusal:
        call(*args, **kwargs)
    message = str(refusal.value)
    assert message.startswith("build_static: "), message
    return message


# ------------------------------------------------------------ small parts

def test_replace_once_refuses_an_anchor_that_moved_or_doubled():
    """Every edit to the template is a string match, and a miss is silent
    by default: the page builds without the paragraph that says it is not
    the Flask app. So none and two are both refusals."""
    assert build.replace_once("a b c", "b", "B", "test") == "a B c"
    message = stopped(build.replace_once, "a c", "b", "B", "add the banner")
    assert "could not add the banner" in message and "found 0" in message
    assert "found 2" in stopped(build.replace_once, "b b", "b", "B", "x")


def test_write_creates_the_directory_and_writes_lf_utf8(tmp_path):
    target = tmp_path / "deep" / "er" / "file.js"
    build.write(str(target), "café\nline two\n")
    assert target.read_bytes() == "café\nline two\n".encode("utf-8")
    assert build.read(str(target)) == "café\nline two\n"


def test_the_comparison_corpus_is_the_data_plus_every_awkward_message():
    messages = build.corpus()
    frame = spamlib.load_data()
    assert messages[:len(frame)] == list(frame["text"].astype(str))
    assert messages[len(frame):] == build.ADVERSARIAL
    # The page's own samples are what anyone clicks first.
    assert "Hey are we still on for lunch tomorrow at noon" in messages


def test_the_replayed_requests_include_the_refusals_and_leave_out_retrain():
    """/api/retrain is the one route where Flask and the shim are meant to
    differ, so replaying it would assert a lie."""
    messages = ["first", "second"] + ["more"] * 30
    requests = build.requests_for(messages)
    urls = {url for _, url, _ in requests}
    assert urls == {"/api/model", "/api/classify", "/api/batch"}
    assert requests[0] == ("GET", "/api/model", None)
    assert ("POST", "/api/classify", {"message": "first"}) in requests
    assert ("POST", "/api/classify", {"message": "  second  "}) in requests
    assert ("POST", "/api/batch", {"messages": messages[:25]}) in requests
    over = [body for _, _, body in requests
            if isinstance(body, dict)
            and len(str(body.get("message", ""))) > web.MAX_MESSAGE_CHARS]
    assert over, "nothing exercises the message-length cap"
    assert ("POST", "/api/batch", {"messages": ["x"] * (web.MAX_BATCH + 1)}) \
        in requests, "nothing exercises the batch cap"


def test_differences_names_the_path_to_every_disagreement():
    want = {"a": 1, "b": [1, 2], "c": {"d": True}, "only_py": 0,
            "short": [1, 2, 3]}
    got = {"a": 1.0, "b": [1, 3], "c": {"d": 1}, "only_js": 0,
           "short": [1]}
    found = build.differences(want, got)
    assert found[0] == ".b[1]: python 2, browser 3"
    assert found[1] == ".c.d: python True, browser 1", (
        "true and 1 are the same in Python and must not be here")
    assert found[2] == ".only_js: only the browser has it (0)"
    assert found[3] == ".only_py: only python has it (0)"
    assert found[4].startswith(".short: python has 3 entries, browser has 1")
    assert len(found) == 5, "1 and 1.0 are the same JSON number"
    assert build.differences(3, 4) == [".: python 3, browser 4"]
    assert build.differences([1], [1, 2])[0].startswith(
        ".: python has 1 entries")


def test_report_stops_with_a_count_and_at_most_twenty_five_lines():
    assert build.report("nothing", []) is None

    one = stopped(build.report, "pass", ["only this"])
    assert "disagrees with the Python in 1 place.\n" in one
    assert "Nothing has been written to docs/app." in one
    assert "only this" in one

    many = stopped(build.report, "pass", ["line %d" % i for i in range(30)],
                   "context\n")
    assert "in 30 places." in many and "context\n" in many
    assert "line 24" in many and "line 25" not in many
    assert many.endswith("... and 5 more")


def test_short_keeps_a_build_log_readable():
    assert build.short("free") == "'free'"
    long = build.short("x" * 200)
    assert len(long) == 71 and long.endswith("...'")


def test_the_build_banner_states_the_model_it_was_built_from():
    text = build.banner({"chosen": "Complement Naive Bayes",
                         "threshold": 0.52381, "terms": ["t"] * 4479})
    assert "Complement Naive Bayes at a decision threshold of\n    0.5238" in text
    assert "4,479-term TF-IDF vocabulary" in text
    assert build.REPO in text


def test_sizes_reports_raw_and_gzipped_bytes_per_file():
    rows = build.sizes({"b.js": "x" * 1000, "a.css": "café"})
    assert [row[0] for row in rows] == ["a.css", "b.js"]
    assert rows[0][1] == 5, "bytes, not characters"
    assert rows[1][1] == 1000
    assert rows[1][2] == len(gzip.compress(b"x" * 1000, 9))


def test_the_harness_is_the_published_files_in_load_order():
    written = {"js/model-data.js": "data", "js/spam.js": "spam",
               "js/static-api.js": "api", "js/app.js": "app"}
    assert build.harness(written) == ["data", "spam", "api"]


# ---------------------------------------------------------------- the model

def test_load_model_trains_and_saves_when_nothing_is_on_disk(
        fitted, tmp_path, monkeypatch, capsys):
    """A fresh clone has no .joblib files, and the build must still work."""
    model_path = tmp_path / "m.joblib"
    vector_path = tmp_path / "v.joblib"
    monkeypatch.setattr(spamlib, "MODEL_PATH", str(model_path))
    monkeypatch.setattr(spamlib, "VECTORIZER_PATH", str(vector_path))
    trained = []

    def train(frame, verbose=True):
        trained.append((len(frame), verbose))
        return fitted

    monkeypatch.setattr(spamlib, "train_and_evaluate", train)
    assert build.load_model() == fitted
    assert trained == [(len(spamlib.load_data()), False)]
    assert model_path.exists() and vector_path.exists()
    assert "no saved artifacts; training" in capsys.readouterr().out

    # And the second time round, it loads what the first one saved.
    trained.clear()
    model, vectorizer = build.load_model()
    assert trained == [], "it retrained with a model on disk"
    assert model.threshold == fitted[0].threshold
    assert capsys.readouterr().out == ""


def test_a_naive_bayes_model_is_exported_whole(fitted):
    model, vectorizer = fitted
    payload = build.collect_model(model, vectorizer)
    width = len(vectorizer.get_feature_names_out())

    assert payload["name"] == "Multinomial Naive Bayes"
    assert payload["estimator"] == "MultinomialNB"
    assert payload["chosen"] == "Multinomial Naive Bayes"
    assert payload["threshold"] == model.threshold
    assert payload["kind"] == "naivebayes" and payload["coef"] is None
    assert payload["addPrior"] is True
    assert len(payload["terms"]) == len(payload["idf"]) == width
    assert payload["logProb"] == model.feature_log_prob_.tolist(), (
        "a weight was rounded or reordered on the way out")
    assert payload["logPrior"] == model.class_log_prior_.tolist()
    assert payload["shoutRatio"] == spamlib.SHOUT_RATIO
    assert payload["maxMessageChars"] == web.MAX_MESSAGE_CHARS
    assert payload["maxBatch"] == web.MAX_BATCH


def test_complement_nb_is_exported_without_the_prior(fitted):
    """ComplementNB's joint log-likelihood leaves the prior out, and the
    flag is the one line of difference spam.js needs to know about."""
    _, vectorizer = fitted
    frame = spamlib.load_data()
    X = vectorizer.transform(frame["clean_text"])
    y = frame["label"].map(spamlib.LABELS).to_numpy()
    payload = build.collect_model(ComplementNB().fit(X, y), vectorizer)
    assert payload["kind"] == "naivebayes"
    assert payload["addPrior"] is False
    assert payload["name"] == payload["chosen"] == "ComplementNB"
    assert payload["threshold"] == 0.5, "a bare estimator cuts at 0.5"


def test_a_linear_model_is_exported_as_coefficients(fitted):
    _, vectorizer = fitted
    frame = spamlib.load_data()
    X = vectorizer.transform(frame["clean_text"])
    y = frame["label"].map(spamlib.LABELS).to_numpy()
    estimator = LogisticRegression(max_iter=spamlib.MAX_ITER).fit(X, y)
    payload = build.collect_model(estimator, vectorizer)
    assert payload["kind"] == "linear"
    assert payload["coef"] == estimator.coef_.ravel().tolist()
    assert payload["intercept"] == float(estimator.intercept_[0])
    assert payload["logProb"] is None and payload["logPrior"] is None
    assert payload["addPrior"] is False


def _width(vectorizer):
    return len(vectorizer.get_feature_names_out())


REFUSALS = [
    ("embedding backend",
     lambda v: types.SimpleNamespace(), lambda v: types.SimpleNamespace(
         wants_raw_text=True),
     "trained with the optional embedding backend"),
    ("not tf-idf",
     lambda v: types.SimpleNamespace(), lambda v: CountVectorizer(),
     "expected a TfidfVectorizer, found CountVectorizer"),
    ("configuration",
     lambda v: types.SimpleNamespace(),
     lambda v: TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True),
     "the vectoriser's ngram_range is (1, 1) and spam.js ports (1, 2)"),
    ("no weights",
     lambda v: types.SimpleNamespace(classes_=np.array([0, 1])), None,
     "(SimpleNamespace) exposes neither coef_ nor feature_log_prob_"),
    ("class order",
     lambda v: types.SimpleNamespace(
         feature_log_prob_=np.zeros((2, _width(v))),
         class_log_prior_=np.zeros(2), classes_=np.array([1, 0])), None,
     "spam.js assumes the two-class [ham, spam]"),
    ("three classes",
     lambda v: types.SimpleNamespace(
         feature_log_prob_=np.zeros((3, _width(v))),
         class_log_prior_=np.zeros(3), classes_=np.array([0, 1])), None,
     "feature_log_prob_ has 3 rows, not 2."),
    ("short coef",
     lambda v: types.SimpleNamespace(
         coef_=np.zeros((1, _width(v) - 1)), intercept_=np.zeros(1),
         classes_=np.array([0, 1])), None,
     "coef has %d entries against %d vocabulary terms."),
    ("short log-prob row",
     lambda v: types.SimpleNamespace(
         feature_log_prob_=np.zeros((2, _width(v) - 1)),
         class_log_prior_=np.zeros(2), classes_=np.array([0, 1])), None,
     "feature_log_prob_ row has %d entries against %d vocabulary terms."),
]


@pytest.mark.parametrize("make_model, make_vectorizer, expected",
                         [case[1:] for case in REFUSALS],
                         ids=[case[0] for case in REFUSALS])
def test_a_model_spam_js_cannot_score_is_refused(fitted, make_model,
                                                 make_vectorizer, expected):
    """Each of these would otherwise ship plausible numbers computed with
    the wrong arithmetic -- the failure this build exists to prevent."""
    _, real = fitted
    vectorizer = real if make_vectorizer is None else make_vectorizer(real)
    message = stopped(build.collect_model, make_model(real), vectorizer)
    if "%d" in expected:
        width = _width(real)
        expected = expected % (width - 1, width)
    assert expected in message


def test_the_metrics_shipped_are_measured_the_way_the_app_measures_them(
        fitted):
    model, vectorizer = fitted
    measured, size = build.model_metrics(model, vectorizer)
    frame = spamlib.load_data()
    assert size == len(frame)
    assert measured == web.evaluate(frame, model, vectorizer)


def test_the_data_file_round_trips_every_number_and_is_ascii():
    """A `<script src>` carries no charset, so a non-ASCII term decoded as
    Latin-1 would never match. And nothing may be rounded."""
    payload = {"terms": ["café", "free"], "idf": [1.0 / 3, 2.5],
               "threshold": 0.45763718, "metrics": {"testSize": 412}}
    text = build.data_file(payload)
    assert text.isascii()
    assert "out-of-fold over 412 messages" in text
    body = text.split("const SPAM_MODEL_DATA = ", 1)[1].rstrip().rstrip(";")
    assert json.loads(body) == payload


# ---------------------------------------------------------------- the browser

def test_without_playwright_the_build_says_how_to_get_it(monkeypatch):
    """There is no --skip-checks flag, so the refusal has to say what to
    install. None in sys.modules makes the import fail whether or not
    Playwright is installed on the machine running this."""
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(SystemExit) as refusal:
        with build.chromium():
            pass                                      # never reached
    assert "pip install playwright" in str(refusal.value)
    assert "no --skip-checks flag on purpose" in str(refusal.value)


def test_the_system_chrome_is_used_when_it_is_there(tmp_path, monkeypatch):
    browser = FaithfulPort(synthetic())
    launches = install_playwright(monkeypatch, browser)

    chrome = tmp_path / "chrome.exe"
    chrome.write_bytes(b"")
    monkeypatch.setattr(build, "CHROME", str(chrome))
    with build.chromium() as instance:
        assert instance is browser
    assert launches == [{"executable_path": str(chrome)}]
    assert browser.closed

    monkeypatch.setattr(build, "CHROME", str(tmp_path / "absent.exe"))
    with build.chromium():
        pass
    assert launches[-1] == {}, "Playwright's own Chromium, with no path"


def test_the_browser_is_closed_even_when_a_check_fails(monkeypatch):
    browser = FaithfulPort(synthetic())
    install_playwright(monkeypatch, browser)
    with pytest.raises(SystemExit):
        with build.chromium():
            build.stop("a check failed")
    assert browser.closed


def test_a_bundle_that_throws_on_load_is_refused(monkeypatch):
    browser = FaithfulPort(synthetic(), load_errors=["SyntaxError: bad"])
    with pytest.raises(SystemExit) as refusal:
        with build.loaded(browser, ["one();", "two();"]):
            pass                                      # never reached
    assert "does not load" in str(refusal.value)
    assert "SyntaxError: bad" in str(refusal.value)
    assert browser.pages[0].closed


def test_a_loaded_bundle_holds_every_source_in_order_and_ignores_console_logs():
    browser = FaithfulPort(synthetic())
    with build.loaded(browser, ["first();", "second();"]) as (page, problems):
        assert problems == [], "a console.log was counted as an error"
        assert page.html.index("first();") < page.html.index("second();")
        assert page.html.count("<script>") == 2
        assert '<meta charset="utf-8">' in page.html
    assert page.closed


# ---------------------------------------------------------- the four passes

class Answers:
    """A page that answers each script with a fixed value."""

    def __init__(self, **by_name):
        self.by_script = {getattr(build, name): value
                          for name, value in by_name.items()}

    def evaluate(self, script, arg=None):
        return self.by_script[script]


def test_pass_zero_returns_the_cut_and_the_nearest_scores_either_side():
    cut, under, over = build.check_threshold(
        Answers(THRESHOLD_JS=0.45), synthetic())
    assert cut == 0.45
    assert under == pytest.approx(0.25)
    assert over == pytest.approx(0.02)


def test_pass_zero_compares_the_cut_bit_for_bit():
    message = stopped(build.check_threshold,
                      Answers(THRESHOLD_JS=0.45 + 1e-12), synthetic())
    assert ("published decision threshold is %.17g and the fitted model's "
            "is %.17g" % (0.45 + 1e-12, 0.45)) in message


def test_pass_zero_refuses_a_corpus_entirely_on_one_side_of_the_cut():
    reference = synthetic(probabilities=(0.6, 0.7))
    message = stopped(build.check_threshold, Answers(THRESHOLD_JS=0.45),
                      reference)
    assert "same side" in message


def test_pass_zero_refuses_a_corpus_with_nothing_between_the_cut_and_half():
    """Without such a message a port that ignored the exported cut and used
    0.5 would agree with the Python on every verdict."""
    reference = synthetic(probabilities=(0.2, 0.9))
    message = stopped(build.check_threshold, Answers(THRESHOLD_JS=0.45),
                      reference)
    assert "between the chosen cut (0.4500) and the 0.5" in message


def test_a_cut_above_half_needs_no_band_below_it():
    """The band check is for a cut under 0.5; over it, a straddled corpus is
    enough."""
    reference = synthetic(threshold=0.6, probabilities=(0.2, 0.9))
    assert build.check_threshold(Answers(THRESHOLD_JS=0.6),
                                 reference)[0] == 0.6


def test_pass_one_compares_cleaned_text_exactly():
    reference = synthetic()
    assert build.check_cleaning(Answers(CLEAN_JS=reference["cleaned"]),
                                reference) == 3
    browser = list(reference["cleaned"])
    browser[1] = "message  1"
    message = stopped(build.check_cleaning, Answers(CLEAN_JS=browser),
                      reference)
    assert "in 1 place." in message
    assert "'message 1'\n      python  'message 1'\n      browser " \
           "'message  1'" in message
    assert "what \\w, \\d, \\s or \\b mean" in message


def _classified(reference):
    out = copy.deepcopy(reference["payloads"])
    for payload, probability in zip(out, reference["probabilities"]):
        payload["probability"] = probability
    return out


def test_pass_two_agrees_with_a_faithful_port():
    reference = synthetic()
    assert build.check_scoring(Answers(CLASSIFY_JS=_classified(reference)),
                               reference) == 3


def test_pass_two_catches_every_kind_of_disagreement():
    reference = synthetic()
    reference["verdicts"][0] = "spam"             # the Python with itself
    browser = _classified(reference)
    browser[1]["probability"] += 1e-9              # P(spam)
    browser[1]["confidence"] = 0.5                 # a displayed field
    browser[2]["tokens"][0]["weight"] = 0.2        # a token chip
    message = stopped(build.check_scoring, Answers(CLASSIFY_JS=browser),
                      reference)
    assert "in 4 places." in message
    assert "predict_message says 'spam' and app.classify says 'ham'" in message
    assert "P(spam) python 0.46999999999999997, browser" in message
    assert "confidence python 0.47, browser 0.5" in message
    assert "tokens[0].weight: python 0.1, browser 0.2" in message


def test_pass_three_replays_the_routes_and_compares_status_and_body():
    reference = synthetic()
    assert build.check_routes(Answers(ROUTES_JS=reference["replies"]),
                              reference) == 2
    browser = copy.deepcopy(reference["replies"])
    browser[0]["status"] = 500
    browser[1]["json"]["label"] = "spam"
    message = stopped(build.check_routes, Answers(ROUTES_JS=browser),
                      reference)
    assert "GET /api/model.status: python 200, browser 500" in message
    assert ('POST /api/classify \'{"message": "message 0"}\'.json.label: '
            "python 'ham', browser 'spam'") in message


def test_run_checks_reports_each_pass_when_loud_and_nothing_when_quiet(
        capsys):
    reference = synthetic()
    written = {"js/model-data.js": "d", "js/spam.js": "s",
               "js/static-api.js": "a"}
    build.run_checks(FaithfulPort(reference), written, reference)
    printed = capsys.readouterr().out
    assert "threshold  == threshold            0.45000000000000001" in printed
    assert "nearest below 0.2500 under, nearest above 0.0200 over" in printed
    assert "clean_text == cleanText            3 messages" in printed
    assert "classify   == classify             3 messages" in printed
    assert "app.py     == static-api.js        2 requests" in printed

    build.run_checks(FaithfulPort(reference), written, reference, loud=False)
    assert capsys.readouterr().out == ""


def test_errors_logged_during_the_checks_fail_the_build():
    """Every comparison can agree and the page still be broken: an error
    thrown in an event handler is logged, not returned."""
    reference = synthetic()
    written = {"js/model-data.js": "d", "js/spam.js": "s",
               "js/static-api.js": "a"}
    browser = FaithfulPort(reference, check_errors=["TypeError: x is null"])
    message = stopped(build.run_checks, browser, written, reference, False)
    assert "logged errors while being checked" in message
    assert "TypeError: x is null" in message


# ------------------------------------------------------------------- --prove

def test_prove_catches_all_six_sabotages(capsys):
    reference = synthetic()
    written = {"js/model-data.js": "const SPAM_MODEL_DATA = {};",
               "js/spam.js": "s", "js/static-api.js": "a"}
    build.prove(FaithfulPort(reference), written, reference)
    printed = capsys.readouterr().out
    for name, _, description in build.SABOTAGE:
        assert "%-12s %-36s caught" % (name, description) in printed, name
    assert "all 6 caught" in printed
    # The published bundle itself is left as it was.
    assert written["js/model-data.js"] == "const SPAM_MODEL_DATA = {};"


def test_prove_fails_the_build_when_a_sabotage_goes_unnoticed():
    """The point of --prove: a comparison that cannot fail reads like one
    that passed. A port blind to two of the breakages is a hole, and the
    build says which."""
    reference = synthetic()
    written = {"js/model-data.js": "d", "js/spam.js": "s",
               "js/static-api.js": "a"}
    message = stopped(build.prove, FaithfulPort(reference,
                                                blind={"idf", "vocabulary"}),
                      written, reference)
    assert "went undetected" in message
    assert "idf (one idf weight out by a millionth)" in message
    assert "vocabulary (one vocabulary term misspelt)" in message
    assert "logprob" not in message


# ------------------------------------------------------------------ the page

def test_the_page_is_the_flask_template_with_the_server_taken_out():
    payload = {"chosen": "Multinomial Naive Bayes", "threshold": 0.4576,
               "terms": ["t"] * 10}
    html = build.page_html(build.read(TEMPLATE), payload)
    assert "<title>Spam Classifier — browser build</title>" in html
    assert '<link rel="stylesheet" href="css/style.css">' in html
    assert '<link rel="stylesheet" href="css/static.css">' in html
    assert "This is the browser-only build." in html
    assert "Inference only: this page carries the numbers" in html
    assert "{{" not in html and "{%" not in html
    positions = [html.index('<script src="%s"></script>' % name)
                 for name in build.SCRIPT_ORDER]
    assert positions == sorted(positions), "the scripts are out of order"


def test_jinja_left_in_the_page_is_refused():
    payload = {"chosen": "NB", "threshold": 0.5, "terms": []}
    template = build.read(TEMPLATE).replace(
        "</footer>", "{{ request.path }}</footer>", 1)
    message = stopped(build.page_html, template, payload)
    assert "unresolved Jinja left in the page: ['{{ request.path }}']" in message


def test_the_bundle_copies_the_flask_files_byte_for_byte():
    payload = {"chosen": "NB", "threshold": 0.5, "terms": ["a"],
               "metrics": {"testSize": 1}}
    written = build.collect(payload)
    assert sorted(written) == ["css/static.css", "css/style.css",
                               "index.html", "js/app.js", "js/model-data.js",
                               "js/spam.js", "js/static-api.js",
                               "js/static-ui.js"]
    app_js = build.read(os.path.join(ROOT, "static", "js", "app.js"))
    assert written["js/app.js"] == build.BANNER % "static/js/app.js" + app_js
    for name in build.STATIC_JS:
        assert written["js/" + name] == build.read(
            os.path.join(build.SRC, "js", name))


def test_prune_removes_only_what_the_build_did_not_write(tmp_path,
                                                         monkeypatch, capsys):
    out = tmp_path / "app"
    monkeypatch.setattr(build, "OUT", str(out))
    build.prune({"index.html"})                    # no directory yet: no-op
    assert not out.exists()

    (out / "js").mkdir(parents=True)
    (out / "index.html").write_text("kept", encoding="utf-8")
    (out / "js" / "renamed.js").write_text("stale", encoding="utf-8")
    build.prune({"index.html"})
    assert (out / "index.html").exists()
    assert not (out / "js" / "renamed.js").exists()
    assert "removed stale %s" % os.path.join("js", "renamed.js") in \
        capsys.readouterr().out


# ---------------------------------------------------------------------- main

def test_an_unknown_option_is_refused_before_anything_is_built(monkeypatch):
    monkeypatch.setattr(build, "load_model",
                        lambda: pytest.fail("it started building"))
    message = stopped(build.main, ["build_static.py", "--skip-checks"])
    assert "unknown option '--skip-checks'. The only one is --prove." in message


def _published(tmp_path, monkeypatch, fitted, argv):
    """Run the whole build into tmp_path with the faithful browser."""
    out = tmp_path / "app"
    (out / "js").mkdir(parents=True)
    (out / "js" / "old.js").write_text("from an older build", encoding="utf-8")
    monkeypatch.setattr(build, "OUT", str(out))
    monkeypatch.setattr(build, "CHROME", str(tmp_path / "no-chrome.exe"))

    browsers = []
    real_reference = build.python_reference

    def reference(messages, model, vectorizer):
        found = real_reference(messages, model, vectorizer)
        browsers.append(FaithfulPort(found))
        return found

    monkeypatch.setattr(build, "python_reference", reference)

    @contextlib.contextmanager
    def sync_playwright():
        class Chromium:
            def launch(self, **options):
                assert options == {}
                return browsers[0]
        yield types.SimpleNamespace(chromium=Chromium())

    api = types.ModuleType("playwright.sync_api")
    api.sync_playwright = sync_playwright
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)

    build.main(argv)
    return out, browsers[0]


def test_the_build_checks_the_port_then_writes_docs_app(
        artifacts, fitted, tmp_path, monkeypatch, capsys):
    """End to end, with the real model export, the real Python reference
    and the real Flask replies -- only the browser is a stand-in."""
    model, vectorizer = fitted
    out, browser = _published(tmp_path, monkeypatch, fitted,
                              ["build_static.py"])
    printed = capsys.readouterr().out
    messages = len(build.corpus())

    assert "exporting Multinomial Naive Bayes at threshold %.4f, %d tf-idf " \
           "terms" % (model.threshold, _width(vectorizer)) in printed
    assert "over %d messages..." % messages in printed
    assert "clean_text == cleanText            %d messages" % messages in printed
    assert "proving" not in printed, "--prove ran without being asked"
    assert "wrote 8 files to docs/app" in printed
    assert "removed stale %s" % os.path.join("js", "old.js") in printed
    assert browser.closed

    published = sorted(os.path.relpath(os.path.join(here, name), str(out))
                       .replace("\\", "/")
                       for here, _, names in os.walk(str(out))
                       for name in names)
    assert published == ["css/static.css", "css/style.css", "index.html",
                         "js/app.js", "js/model-data.js", "js/spam.js",
                         "js/static-api.js", "js/static-ui.js"]

    data = build.read(str(out / "js" / "model-data.js"))
    payload = json.loads(data.split("const SPAM_MODEL_DATA = ", 1)[1]
                         .rstrip().rstrip(";"))
    assert payload["threshold"] == model.threshold
    assert payload["datasetSize"] == len(spamlib.load_data())
    assert payload["metrics"]["testSize"] == payload["datasetSize"]
    assert payload["terms"] == list(vectorizer.get_feature_names_out())
    assert "This is the browser-only build." in build.read(
        str(out / "index.html"))


def test_the_build_with_prove_breaks_the_real_bundle_six_ways(
        artifacts, fitted, tmp_path, monkeypatch, capsys):
    _published(tmp_path, monkeypatch, fitted, ["build_static.py", "--prove"])
    printed = capsys.readouterr().out
    assert "proving the comparison is not vacuous..." in printed
    assert "all 6 caught" in printed
    assert "wrote 8 files to docs/app" in printed
