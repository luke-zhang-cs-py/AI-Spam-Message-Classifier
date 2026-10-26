"""tools/refresh_figures.py, the half of the published-figures check that writes.

`test_published_figures.py` reads docs/index.html and says when a number on it
is wrong; this script is what puts the right one there. It was omitted from
coverage as a developer script, which meant the tool that keeps the page
honest was the one thing in the project nobody had checked.

It shells out twice -- a full `pytest --cov` run and a `--collect-only` --
and both are replaced here with stand-ins that hand back a known answer. A
test that ran the real suite from inside the suite would recurse, and would
take as long as the suite. Every file it reads or writes is redirected into
`tmp_path`: the real page and README are never touched.
"""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import refresh_figures as figures                         # noqa: E402

RUNNING = "%d.%d" % sys.version_info[:2]


def page_with(modules, tests, test_lines=7, measured_with="Python 3.0"):
    """A page holding just the blocks the script rewrites, in its layout."""
    module_rows = ",\n".join(
        "  { name: '%s', lines: %d, stmts: %d, missed: %d, does: '%s' }"
        % row for row in modules)
    test_rows = ",\n".join("  { file: '%s', n: %d, of: '%s' }" % row
                           for row in tests)
    return ("<script>\nvar MODULES = [\n%s\n];\n\nvar TESTS = [\n%s\n];\n\n"
            "var TEST_LINES = %d;\nvar MEASURED_WITH = '%s';\n</script>\n"
            % (module_rows, test_rows, test_lines, measured_with))


class Finished:
    """What `subprocess.run` hands back, as far as the script reads it."""

    def __init__(self, stdout=""):
        self.stdout = stdout
        self.returncode = 0


# ------------------------------------------------------------------ measure

def test_measure_reads_the_json_report_the_coverage_run_writes(tmp_path,
                                                               monkeypatch):
    """The figures come from coverage's own JSON report, keyed by file, with
    Windows separators turned into the forward slashes the page uses -- and
    the line count from the file itself, not from the report."""
    monkeypatch.setattr(figures, "ROOT", str(tmp_path))
    (tmp_path / "app.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    # A backslashed name, as coverage reports it on Windows. Built with
    # os.path.join so the file it names exists on either platform.
    windows_name = "tools\\build.py"
    target = os.path.join(str(tmp_path), windows_name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("x = 1\n")

    seen = {}

    def run(args, **options):
        seen["args"], seen["options"] = args, options
        report_path = [a for a in args if a.startswith("--cov-report=json:")]
        assert report_path, "the run did not ask coverage for a JSON report"
        seen["report"] = report_path[0].split(":", 1)[1]
        with open(seen["report"], "w", encoding="utf-8") as handle:
            json.dump({"files": {
                "app.py": {"summary": {"num_statements": 3,
                                       "missing_lines": 1}},
                windows_name: {"summary": {"num_statements": 1,
                                           "missing_lines": 0}},
            }}, handle)
        return Finished()

    monkeypatch.setattr(figures.subprocess, "run", run)
    measured = figures.measure()

    assert measured == {
        "app.py": {"lines": 3, "stmts": 3, "missed": 1},
        "tools/build.py": {"lines": 1, "stmts": 1, "missed": 0},
    }
    assert seen["args"][:4] == [sys.executable, "-m", "pytest", "-q"]
    assert "--cov" in seen["args"]
    assert seen["options"]["cwd"] == str(tmp_path)
    assert not os.path.exists(seen["report"]), "the temporary report was left behind"


# ------------------------------------------------------------------ collect

def test_collect_counts_tests_per_file_from_a_collect_only_run(monkeypatch):
    """By file basename, because that is how the page's TESTS block names
    them; the summary line has no `::` and is not a test."""
    listing = ("tests/test_a.py::test_one\n"
               "tests/test_a.py::test_two[param]\n"
               "tests/deeper/test_b.py::test_three\n"
               "\n3 tests collected in 0.01s\n")
    monkeypatch.setattr(figures.subprocess, "run",
                        lambda args, **options: Finished(listing))
    assert figures.collect() == {"test_a.py": 2, "test_b.py": 1}


def test_a_collection_that_finds_nothing_stops_instead_of_writing_zeros(
        monkeypatch):
    """An import error in a test file makes `--collect-only` print the error
    and no test ids. Writing that page would set every count to nothing."""
    monkeypatch.setattr(figures.subprocess, "run",
                        lambda args, **options: Finished("ERROR collecting"))
    with pytest.raises(SystemExit) as stopped:
        figures.collect()
    assert "collection produced nothing" in str(stopped.value)
    assert "ERROR collecting" in str(stopped.value)


# -------------------------------------------------------------- fix_modules

def test_module_figures_are_rewritten_sorted_and_the_change_reported():
    page = page_with([("small.py", 5, 2, 0, "the small one"),
                      ("big.py", 50, 20, 3, "the big one")],
                     [("test_a.py", 1, "a")])
    changes = []
    fixed = figures.fix_modules(page, {
        "small.py": {"lines": 5, "stmts": 2, "missed": 0},
        "big.py": {"lines": 60, "stmts": 25, "missed": 0},
    }, changes)

    assert changes == ["big.py                   (50, 20, 3) -> (60, 25, 0)"], (
        "only the row that moved should be reported")
    block = re.search(r"var MODULES = \[\n(.*?)\n\];", fixed, re.S).group(1)
    assert block.splitlines() == [
        "  { name: 'big.py',   lines:  60, stmts:  25, missed:  0, "
        "does: 'the big one' },",
        "  { name: 'small.py', lines:   5, stmts:   2, missed:  0, "
        "does: 'the small one' }",
    ], "largest first, padded to one column, no comma after the last row"


def test_a_module_with_no_description_stops_the_rewrite():
    """Only a person can say what a module is for, so the script refuses
    rather than inventing a row."""
    page = page_with([("old.py", 5, 2, 0, "old")], [("test_a.py", 1, "a")])
    with pytest.raises(SystemExit) as stopped:
        figures.fix_modules(page, {
            "old.py": {"lines": 5, "stmts": 2, "missed": 0},
            "new.py": {"lines": 9, "stmts": 4, "missed": 0},
        }, [])
    assert "no description on the page for new.py" in str(stopped.value)


def test_a_module_coverage_stopped_reporting_is_flagged_not_dropped_silently():
    page = page_with([("kept.py", 5, 2, 0, "kept"),
                      ("gone.py", 5, 2, 0, "gone")], [("test_a.py", 1, "a")])
    changes = []
    figures.fix_modules(page, {"kept.py": {"lines": 5, "stmts": 2,
                                           "missed": 0}}, changes)
    assert changes == ["gone.py is on the page but coverage no longer "
                       "reports it -- remove it, or check whether it "
                       "became omitted"]


# ---------------------------------------------------------------- fix_tests

def test_test_counts_are_rewritten_and_a_file_that_collects_nothing_flagged():
    page = page_with([("a.py", 1, 1, 0, "a")],
                     [("test_a.py", 3, "the a tests"),
                      ("test_b.py", 9, "the b tests"),
                      ("test_gone.py", 2, "removed")])
    changes = []
    fixed = figures.fix_tests(page, {"test_a.py": 4, "test_b.py": 9}, changes)

    assert changes == ["test_gone.py is on the page but collects nothing",
                       "test_a.py                    3 -> 4"]
    block = re.search(r"var TESTS = \[\n(.*?)\n\];", fixed, re.S).group(1)
    assert block.splitlines() == [
        "  { file: 'test_b.py', n:  9, of: 'the b tests' },",
        "  { file: 'test_a.py', n:  4, of: 'the a tests' }",
    ]


def test_a_test_file_with_no_description_stops_the_rewrite():
    page = page_with([("a.py", 1, 1, 0, "a")], [("test_a.py", 1, "a")])
    with pytest.raises(SystemExit) as stopped:
        figures.fix_tests(page, {"test_a.py": 1, "test_new.py": 2}, [])
    assert "no description on the page for test_new.py" in str(stopped.value)


def test_the_published_page_is_a_fixed_point_of_its_own_figures():
    """The real docs/index.html, rewritten with the figures it already
    states, must come back byte for byte. A script that reformats the file on
    every run leaves a diff nobody asked for, and the docstring promises it
    does not."""
    page = figures.read(figures.PAGE)
    stated_modules = {
        name: {"lines": int(lines), "stmts": int(stmts),
               "missed": int(missed)}
        for name, lines, stmts, missed in re.findall(
            r"\{ name: '([\w./]+)',\s*lines:\s*(\d+),\s*stmts:\s*(\d+),"
            r"\s*missed:\s*(\d+),", page)}
    stated_tests = {name: int(n) for name, n in re.findall(
        r"\{ file: '([\w.]+)',\s*n:\s*(\d+),", page)}
    assert stated_modules and stated_tests, "the blocks could not be read"

    changes = []
    again = figures.fix_tests(figures.fix_modules(page, stated_modules,
                                                  changes),
                              stated_tests, changes)
    assert changes == []
    assert again == page


# -------------------------------------------------------- fix_measured_with

def test_the_interpreter_that_measured_the_page_is_recorded():
    """Statement counts differ between Pythons, so the page says which one
    produced them -- and a run under another one says it changed."""
    changes = []
    fixed = figures.fix_measured_with(
        "var MEASURED_WITH = 'Python 2.7';", changes)
    assert fixed == "var MEASURED_WITH = 'Python %s';" % RUNNING
    assert changes == ["MEASURED_WITH Python 2.7 -> %s" % RUNNING]

    unchanged = []
    figures.fix_measured_with(fixed, unchanged)
    assert unchanged == [], "the same interpreter is not a change"


def test_a_page_with_no_measured_with_line_is_refused():
    with pytest.raises(SystemExit) as stopped:
        figures.fix_measured_with("var MODULES = [];", [])
    assert "no MEASURED_WITH line" in str(stopped.value)


# ----------------------------------------------------------- fix_test_lines

def test_the_test_line_total_counts_python_files_under_tests_only(
        tmp_path, monkeypatch):
    """Nested directories count; bytecode caches and non-Python files do
    not."""
    monkeypatch.setattr(figures, "ROOT", str(tmp_path))
    tests = tmp_path / "tests"
    (tests / "sub").mkdir(parents=True)
    (tests / "__pycache__").mkdir()
    (tests / "test_a.py").write_text("1\n2\n3\n", encoding="utf-8")
    (tests / "sub" / "test_b.py").write_text("1\n2\n", encoding="utf-8")
    (tests / "__pycache__" / "cached.py").write_text("1\n" * 50,
                                                     encoding="utf-8")
    (tests / "notes.txt").write_text("1\n" * 50, encoding="utf-8")

    changes = []
    fixed = figures.fix_test_lines("var TEST_LINES = 1;", changes)
    assert fixed == "var TEST_LINES = 5;"
    assert changes == ["TEST_LINES 1 -> 5"]

    unchanged = []
    assert figures.fix_test_lines(fixed, unchanged) == fixed
    assert unchanged == []

    absent = []
    assert figures.fix_test_lines("no such line", absent) == "no such line"
    assert absent == [], "a page without the line has nothing to report"


# --------------------------------------------------------------- fix_readme

def test_the_readmes_figure_sentence_is_rewritten_when_stale(tmp_path,
                                                             monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text("Intro.\n\n9 tests, 50% of 10 statements, checked.\n",
                      encoding="utf-8")
    monkeypatch.setattr(figures, "README", str(readme))

    measured = {"a.py": {"lines": 1, "stmts": 1500, "missed": 15},
                "b.py": {"lines": 1, "stmts": 500, "missed": 0}}
    changes = []
    figures.fix_readme(measured, {"test_a.py": 1200, "test_b.py": 34},
                       changes)

    wanted = "1,234 tests, 99% of 2,000 statements"
    assert readme.read_text(encoding="utf-8") == (
        "Intro.\n\n%s, checked.\n" % wanted)
    assert changes == ["README  9 tests, 50% of 10 statements -> " + wanted]

    again = []
    figures.fix_readme(measured, {"test_a.py": 1200, "test_b.py": 34}, again)
    assert again == [], "a README that is already right is not a change"


def test_a_readme_that_is_absent_or_states_no_figures_is_left_alone(
        tmp_path, monkeypatch):
    measured = {"a.py": {"lines": 1, "stmts": 10, "missed": 0}}

    monkeypatch.setattr(figures, "README", str(tmp_path / "missing.md"))
    figures.fix_readme(measured, {"test_a.py": 1}, [])
    assert not (tmp_path / "missing.md").exists(), "it created a README"

    plain = tmp_path / "README.md"
    plain.write_text("No numbers here.\n", encoding="utf-8")
    monkeypatch.setattr(figures, "README", str(plain))
    changes = []
    figures.fix_readme(measured, {"test_a.py": 1}, changes)
    assert plain.read_text(encoding="utf-8") == "No numbers here.\n"
    assert changes == []


# --------------------------------------------------------------------- main

@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A page and a README in tmp_path, and the two runs replaced."""
    page = tmp_path / "index.html"
    readme = tmp_path / "README.md"
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("1\n2\n3\n4\n", encoding="utf-8")
    monkeypatch.setattr(figures, "ROOT", str(tmp_path))
    monkeypatch.setattr(figures, "PAGE", str(page))
    monkeypatch.setattr(figures, "README", str(readme))
    monkeypatch.setattr(figures, "measure", lambda: {
        "a.py": {"lines": 12, "stmts": 8, "missed": 2}})
    monkeypatch.setattr(figures, "collect", lambda: {"test_a.py": 3})
    return page, readme


def test_main_rewrites_a_stale_page_and_readme_and_says_what_moved(
        staged, capsys):
    page, readme = staged
    page.write_text(page_with([("a.py", 10, 8, 0, "the module")],
                              [("test_a.py", 1, "its tests")],
                              test_lines=1), encoding="utf-8")
    readme.write_text("1 tests, 100% of 8 statements\n", encoding="utf-8")

    figures.main()

    written = page.read_text(encoding="utf-8")
    assert "lines:  12, stmts:   8, missed:  2" in written
    assert "n:  3," in written
    assert "var TEST_LINES = 4;" in written
    assert "var MEASURED_WITH = 'Python %s';" % RUNNING in written
    assert readme.read_text(encoding="utf-8") == (
        "3 tests, 75% of 8 statements\n")

    printed = capsys.readouterr().out
    assert "(10, 8, 0) -> (12, 8, 2)" in printed
    assert "README  1 tests, 100% of 8 statements -> 3 tests, 75% of 8 " \
           "statements" in printed
    assert printed.rstrip().endswith("3 tests, 8 statements, 75% covered")


def test_main_leaves_a_page_that_is_already_right_untouched(
        staged, monkeypatch, capsys):
    page, _ = staged
    # The script's own output for these figures, so the page is right in
    # layout as well as in numbers.
    right = figures.rewrite(page_with([("a.py", 1, 1, 1, "the module")],
                                      [("test_a.py", 1, "its tests")]),
                            figures.measure(), figures.collect(), [])
    page.write_text(right, encoding="utf-8")
    writes = []
    monkeypatch.setattr(figures, "write",
                        lambda path, text: writes.append(path))

    figures.main()

    assert writes == [], "a page that was already right was rewritten"
    assert capsys.readouterr().out == "  3 tests, 8 statements, 75% covered\n"


def test_main_refuses_to_write_output_a_second_pass_would_change(
        staged, monkeypatch):
    """The idempotence check. A rewrite that does not settle would leave a
    diff behind on every run, so nothing is written at all."""
    page, _ = staged
    original = page_with([("a.py", 10, 8, 0, "the module")],
                         [("test_a.py", 1, "its tests")])
    page.write_text(original, encoding="utf-8")
    passes = []

    def drifting(text, measured, counts, changes):
        passes.append(1)
        return text + "<!-- pass %d -->" % len(passes)

    monkeypatch.setattr(figures, "rewrite", drifting)
    with pytest.raises(SystemExit) as stopped:
        figures.main()
    assert "not idempotent" in str(stopped.value)
    assert page.read_text(encoding="utf-8") == original
