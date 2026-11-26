"""The colour themes: complete, readable, dark with no JavaScript, and remembered.

Three pages carry the token system. The Flask page (templates/index.html with
static/css/style.css and static/js/app.js) and its browser build in docs/app/,
which tools/build_static.py copies from those same files, share one set of
tokens; the write-up, docs/index.html, has its own block, the one every
project page built from that template shares.

Each page opens dark from a class written into its markup, steps through a
THEMES list with one button, and keeps the choice under localStorage 'theme'.
What can break, and what each test here is for:

  * a theme added to THEMES without a block, or a block missing a token, so
    one theme silently inherits another's colour (test 1);
  * a token value changed until some pair the page really uses drops under
    WCAG AA -- checked from the tokens, so a pair that only shows in an error
    box, a spam chip or a hover is covered without a browser having to
    happen upon it (test 2);
  * the markup losing its class, so the page is unthemed without JavaScript,
    or the system colour scheme creeping back in (test 3);
  * the toggle not switching, not remembering, or adopting a saved theme
    that no longer exists (test 4);
  * text the token matrix does not know about, read off the rendered page
    in every state it has, in both themes (test 5);
  * a banned or remote font creeping back into a stack or a <link>, or the
    self-hosted faces failing to load (tests 3 and 5).

The pure-Python tests run everywhere, CI included. The browser tests need
Playwright and a Chromium, which CI does not install, so they skip there with
a reason, the same way tools/build_static.py's own browser checks are run by
hand rather than in CI.
"""
import io
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP_CSS = os.path.join(ROOT, "static", "css", "style.css")
APP_JS = os.path.join(ROOT, "static", "js", "app.js")
TEMPLATE = os.path.join(ROOT, "templates", "index.html")
STATIC_CSS = os.path.join(ROOT, "tools", "static_src", "css", "static.css")
DEMO = os.path.join(ROOT, "docs", "app", "index.html")
WRITEUP = os.path.join(ROOT, "docs", "index.html")

# The contract: every theme block in every page sets at least these.
CONTRACT = {
    "--color-bg", "--color-surface", "--color-surface-subtle", "--color-border",
    "--color-border-strong", "--color-text-primary", "--color-text-muted",
    "--color-text-subtle", "--color-accent", "--color-accent-hover",
    "--color-accent-subtle", "--color-text-on-accent", "--color-danger",
}

# The pairs each page actually draws, as (foreground, background) token names
# without the --color- prefix. Text at 4.5:1, control boundaries and marks
# that carry meaning at 3:1.
APP_TEXT = [
    ("text-primary", "bg"), ("text-primary", "surface"),
    ("text-primary", "surface-subtle"),             # code in the build banner
    ("text-primary", "accent-subtle-hover"),        # a secondary button, hovered
    ("text-muted", "bg"),                           # the header line, the footer
    ("text-muted", "surface"),                      # labels, notes, samples
    ("text-muted", "border"),                       # the badge before a verdict
    ("text-subtle", "surface"),                     # the placeholder, on the field's surface fill
    ("text-subtle", "surface-subtle"),              # ...and on an inset, should it move back
    ("accent", "surface"),                          # a hovered sample, the banner link
    ("text-on-accent", "accent"),                   # the Classify button
    ("text-on-accent", "accent-hover"),             # ...hovered, and all through the fade:
                                                    # the text colour is the same at both ends
    ("danger", "surface"), ("danger", "danger-subtle"),     # spam: batch row; badge, chip, error
    ("success", "surface"), ("success", "success-subtle"),  # ham: batch row; badge, chip
]
APP_UI = [
    # The textarea's edge and a secondary button's outline, on both sides of
    # the line: the page or a panel outside, the field's surface fill inside.
    ("border-strong", "bg"), ("border-strong", "surface"),
    ("accent", "bg"), ("accent", "surface"),        # the focus ring; the focused field's edge;
                                                    # the Classify button's face
    ("accent", "surface-subtle"),
    ("accent", "border"), ("danger", "border"), ("success", "border"),  # confidence fill on its track
]
WRITEUP_TEXT = [
    (fg, bg) for fg in ("text-primary", "text-muted", "text-subtle")
    for bg in ("bg", "surface", "surface-subtle")
] + [
    ("accent", "bg"), ("accent", "surface"),        # links, step numbers, a pressed button
    ("text-on-accent", "accent"),
    ("success", "surface"), ("danger", "surface"),  # status flags, always with a glyph and a word
]
WRITEUP_UI = [
    ("border-strong", "bg"), ("border-strong", "surface"),  # button and link-button outlines
    ("accent", "bg"), ("accent", "surface"),        # the focus ring, a pressed button's outline
    ("accent", "track"),                            # a bar on its rail
]

PAGES = {
    # name: (file holding the tokens, file holding THEMES, text pairs, UI pairs)
    "app": (APP_CSS, APP_JS, APP_TEXT, APP_UI),
    "writeup": (WRITEUP, WRITEUP, WRITEUP_TEXT, WRITEUP_UI),
}


def read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def css_of(path):
    """The CSS in a file, comments removed: a stylesheet, or a page's <style>."""
    text = read(path)
    if path.endswith(".html"):
        text = "\n".join(re.findall(r"<style>(.*?)</style>", text, re.S))
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def theme_blocks(css):
    """{theme name: {token: value}} for every .theme-<name> block."""
    blocks = {}
    for names, body in re.findall(r"([^{}]*\.theme-[a-z]+[^{}]*)\{([^}]*)\}", css):
        tokens = dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))
        for name in re.findall(r"\.theme-([a-z]+)", names):
            blocks[name] = tokens
    return blocks


def fallback_selector(css):
    """The selector list of the block the dark theme is declared in."""
    match = re.search(r"([^{}]*)\{[^}]*\}", css[css.index(".theme-dark") - 40:])
    return match.group(1) if match else ""


def themes_of(path):
    match = re.search(r"(?:const|var)\s+THEMES\s*=\s*\[([^\]]*)\]", read(path))
    assert match, "no THEMES list in %s" % os.path.relpath(path, ROOT)
    return re.findall(r"'([a-z]+)'", match.group(1))


def luminance(hex_colour):
    value = hex_colour.strip().lstrip("#")
    assert re.fullmatch(r"[0-9a-fA-F]{6}", value), \
        "theme colours are opaque six-digit hex, so contrast can be read from them: %r" % hex_colour
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    high, low = sorted([luminance(a), luminance(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


# ------------------------------------------------------- 1. complete themes

@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_theme_has_a_block_with_the_same_tokens(page):
    css_path, js_path, _, _ = PAGES[page]
    blocks = theme_blocks(css_of(css_path))
    themes = themes_of(js_path)
    assert themes[0] == "dark", "the first theme is the one in the markup, and it is dark"
    assert len(themes) >= 2
    default = set(blocks["dark"])
    assert CONTRACT <= default, "missing from the dark block: %s" % sorted(CONTRACT - default)
    for name in themes:
        assert name in blocks, "THEMES lists %r but there is no .theme-%s block" % (name, name)
        assert set(blocks[name]) == default, (name, sorted(set(blocks[name]) ^ default))


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_dark_block_is_also_the_zero_specificity_fallback(page):
    """An <html> with an unknown theme class still gets a whole palette."""
    selector = fallback_selector(css_of(PAGES[page][0]))
    assert re.search(r":where\(:root\)\s*,\s*\.theme-dark", selector), selector


# --------------------------------------------------------- 2. AA, from tokens

@pytest.mark.parametrize("page", sorted(PAGES))
def test_every_pair_the_page_uses_meets_aa_in_every_theme(page):
    css_path, js_path, text, ui = PAGES[page]
    blocks = theme_blocks(css_of(css_path))
    short = []
    for name in themes_of(js_path):
        tokens = blocks[name]
        for pairs, need in ((text, 4.5), (ui, 3.0)):
            for fg, bg in pairs:
                ratio = contrast(tokens["--color-" + fg], tokens["--color-" + bg])
                if ratio < need:
                    short.append("%s: %s on %s is %.2f:1, needs %.1f" % (name, fg, bg, ratio, need))
    assert not short, short


@pytest.mark.parametrize("path", [APP_CSS, STATIC_CSS, WRITEUP])
def test_component_rules_read_tokens_not_raw_colours(path):
    css = css_of(path)
    components = re.sub(r"[^{}]*\.theme-[a-z]+[^{}]*\{[^}]*\}", "", css)
    raw = re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", components)
    assert not raw, raw


# --------------------------------------------------- 3. dark without script

@pytest.mark.parametrize("path", [TEMPLATE, DEMO, WRITEUP])
def test_the_markup_opens_dark_and_restores_a_saved_theme_before_paint(path):
    page = read(path)
    assert re.search(r'<html lang="en" class="theme-dark">', page)
    head = page[:page.index("</head>")]
    assert "localStorage.getItem('theme')" in head
    assert "classList.replace('theme-dark', 'theme-' + saved)" in head
    assert re.search(r'<button[^>]*id="themeBtn"[^>]*type="button"[^>]*'
                     r'aria-label="Switch to light theme"', page)


# DESIGN.md: never these, not even as a fallback. A stack ends at the generic
# sans-serif or monospace.
BANNED_FONTS = re.compile(r"\bInter\b|Roboto|Open Sans|\bArial\b|system-ui|-apple-system|"
                          r"BlinkMacSystemFont|Segoe UI|Helvetica")
FONT_STACKS = {
    "--font-display": '"Bricolage Grotesque",sans-serif',
    "--font-sans": '"IBM Plex Sans",sans-serif',
    "--font-mono": '"IBM Plex Mono",monospace',
}


@pytest.mark.parametrize("path", [APP_CSS, STATIC_CSS, TEMPLATE, APP_JS, WRITEUP])
def test_the_system_picks_neither_the_colour_scheme_nor_the_type(path):
    """The theme comes from the class, and the type from the self-hosted
    faces: no system colour scheme, no system or banned font, no font server."""
    text = read(path)
    assert "prefers-color-scheme" not in text
    assert "data-theme" not in text
    assert not BANNED_FONTS.findall(text), BANNED_FONTS.findall(text)
    assert "fonts.googleapis.com" not in text and "fonts.gstatic.com" not in text
    # Every family a rule names is one of the three tokens (or inherits one).
    styles = css_of(path) if path.endswith((".css", ".html")) else ""
    for family in re.findall(r"font-family\s*:\s*([^;}]+)", styles):
        assert family.strip() == "inherit" or family.strip().startswith("var(--font-"), family
    if path in (APP_CSS, WRITEUP):
        stacks = {name: re.sub(r",\s+", ",", value.strip()) for name, value in
                  re.findall(r"(--font-[a-z]+)\s*:\s*([^;]+);", css_of(path))}
        assert stacks == FONT_STACKS, stacks
    if path.endswith(".html"):
        head = text[:text.index("</head>")]
        fonts = re.search(r'<link rel="stylesheet" href="([^"]*fonts\.css[^"]*)">', head)
        assert fonts, "the self-hosted fonts.css is not loaded"
        assert head.index(fonts.group(0)) < head.index("<style>" if path == WRITEUP else "css/style.css")


def test_the_published_demo_was_built_from_these_sources():
    """docs/app/ is generated. A token edited in static/ and not rebuilt would
    leave the published demo on the old palette with every test above green."""
    banner = "/* Generated by tools/build_static.py -- edit %s instead. */\n"
    for out, source, name in (("css/style.css", APP_CSS, "static/css/style.css"),
                              ("js/app.js", APP_JS, "static/js/app.js")):
        built = read(os.path.join(ROOT, "docs", "app", *out.split("/")))
        assert built == banner % name + read(source), \
            "docs/app/%s is stale: run python tools/build_static.py" % out
    assert read(os.path.join(ROOT, "docs", "app", "css", "static.css")) == read(STATIC_CSS)
    # The type as well: fonts.css and every face and licence beside it.
    fonts = ["fonts.css"] + ["fonts/" + name for name in
                             sorted(os.listdir(os.path.join(ROOT, "static", "fonts")))]
    assert len(fonts) == 8, fonts
    for name in fonts:
        with open(os.path.join(ROOT, "static", *name.split("/")), "rb") as want,                 open(os.path.join(ROOT, "docs", "app", *name.split("/")), "rb") as got:
            assert got.read() == want.read(), "docs/app/%s is stale" % name


# ------------------------------------------------------------ the browser

SCAN = r"""() => {
  const parse = c => { const n = c.match(/[\d.]+/g).map(Number);
                       return { rgb: n.slice(0, 3).map(v => v / 255), a: n.length > 3 ? n[3] : 1 }; };
  const lum = c => { const [r, g, b] = parse(c).rgb.map(v => v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
                     return .2126 * r + .7152 * g + .0722 * b; };
  const behind = el => { for (let e = el; e; e = e.parentElement) { const bg = getComputedStyle(e).backgroundColor;
                     if (/\d/.test(bg) && parse(bg).a > .5) return bg; } return getComputedStyle(document.body).backgroundColor; };
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim())) continue;
    const s = getComputedStyle(el);
    if (s.visibility === 'hidden' || s.display === 'none' || !el.getClientRects().length) continue;
    if (el.closest('[hidden], noscript, button:disabled')) continue;   // disabled controls are exempt
    const big = parseFloat(s.fontSize) >= 24 || (parseFloat(s.fontSize) >= 18.66 && +s.fontWeight >= 700);
    const [L1, L2] = [lum(s.color), lum(behind(el))].sort((a, b) => b - a);
    const r = (L1 + .05) / (L2 + .05);
    if (r < (big ? 3 : 4.5)) out.push(el.tagName + '.' + el.className + ' ' + r.toFixed(2) + ' "' + el.textContent.trim().slice(0, 30) + '"');
  }
  return [...new Set(out)];
}"""
THEME = r"(document.documentElement.className.match(/(?:^|\s)theme-(\S+)/) || [])[1]"
BODY_BG = "getComputedStyle(document.body).backgroundColor"


def rgb(hex_colour):
    value = hex_colour.strip().lstrip("#")
    return "rgb(%d, %d, %d)" % tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


BROWSER_PAGES = {
    # name: (file, file with its tokens)
    "demo": (DEMO, APP_CSS),
    "writeup": (WRITEUP, WRITEUP),
}


def dark_bg(page):
    return rgb(theme_blocks(css_of(BROWSER_PAGES[page][1]))["dark"]["--color-bg"])


def url(page):
    return "file:///" + BROWSER_PAGES[page][0].replace(os.sep, "/")


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip(
        "playwright.sync_api", reason="Playwright is not installed (CI does not install it)")
    with sync_api.sync_playwright() as pw:
        try:
            instance = pw.chromium.launch()
        except Exception as error:    # the package without its browser
            pytest.skip("no Chromium for Playwright: %s" % str(error).splitlines()[0])
        yield instance
        instance.close()


@pytest.fixture
def open_page(browser):
    contexts = []

    def opener(page, script=True, seed=None, width=1280):
        context = browser.new_context(java_script_enabled=script, color_scheme="light",
                                      viewport={"width": width, "height": 900})
        contexts.append(context)
        tab = context.new_page()
        tab.errors = []
        tab.on("pageerror", lambda e: tab.errors.append(str(e)))
        tab.on("console", lambda m: tab.errors.append(m.text) if m.type == "error" else None)
        if seed is not None:
            tab.add_init_script("localStorage.setItem('theme', %r)" % seed)
        tab.goto(url(page))
        tab.wait_for_timeout(300)
        return tab

    yield opener
    for context in contexts:
        context.close()


@pytest.mark.parametrize("page", sorted(BROWSER_PAGES))
def test_with_javascript_off_the_page_is_dark(open_page, page):
    """With a light system preference, which must not matter."""
    tab = open_page(page, script=False)
    seen = tab.evaluate("[document.documentElement.className, %s, "
                        "getComputedStyle(document.documentElement).colorScheme]" % BODY_BG)
    assert seen == ["theme-dark", dark_bg(page), "dark"]
    assert not tab.evaluate(SCAN)


@pytest.mark.parametrize("page", sorted(BROWSER_PAGES))
def test_the_toggle_switches_and_is_remembered(open_page, page):
    tab = open_page(page)
    assert tab.evaluate(THEME) == "dark"
    assert tab.get_attribute("#themeBtn", "aria-label") == "Switch to light theme"
    tab.click("#themeBtn")
    assert tab.evaluate(THEME) == "light"
    assert tab.get_attribute("#themeBtn", "aria-label") == "Switch to dark theme"
    assert tab.evaluate("localStorage.getItem('theme')") == "light"
    assert tab.evaluate(BODY_BG) != dark_bg(page)
    tab.reload()
    tab.wait_for_timeout(300)
    assert tab.evaluate(THEME) == "light"
    tab.click("#themeBtn")
    assert [tab.evaluate(THEME), tab.evaluate(BODY_BG)] == ["dark", dark_bg(page)]
    assert not tab.errors, tab.errors


@pytest.mark.parametrize("page", sorted(BROWSER_PAGES))
def test_an_unknown_saved_theme_falls_back_to_dark(open_page, page):
    tab = open_page(page, seed="sepia")
    seen = tab.evaluate("[document.documentElement.className.match(/theme-\\S+/g).join(), %s]" % BODY_BG)
    assert seen == ["theme-dark", dark_bg(page)]
    assert tab.get_attribute("#themeBtn", "aria-label") == "Switch to light theme"


def _demo_states(tab):
    """Every state the classifier page can show, with a scan of each."""
    found = []
    tab.click("#classifyBtn")                                  # empty: the error box
    assert tab.is_visible("#err")
    found += tab.evaluate(SCAN)
    tab.click("#samples li >> nth=0")                          # spam
    tab.wait_for_function("document.getElementById('badge').className === 'badge spam'")
    tab.hover("#samples li >> nth=2")
    found += tab.evaluate(SCAN)
    tab.click("#samples li >> nth=1")                          # ham
    tab.wait_for_function("document.getElementById('badge').className === 'badge ham'")
    tab.hover("#classifyBtn")
    found += tab.evaluate(SCAN)
    tab.fill("#input", "Free entry into our weekly draw, text WIN to 80086 now\n"
                       "Hey are we still on for lunch tomorrow at noon")
    tab.click("#batchBtn")                                     # batch, both verdicts
    tab.wait_for_selector("#batchTable td.v.spam")
    tab.wait_for_selector("#batchTable td.v.ham")
    found += tab.evaluate(SCAN)
    return found


def _writeup_states(tab):
    found = tab.evaluate(SCAN)
    tab.click("#tableBtn")
    found += tab.evaluate(SCAN)
    tab.click("#tableBtn")
    return found


# The faces each page draws with once every state has been shown; a face
# the browser could not fetch stays "unloaded" or turns "error".
FACES = r"""() => document.fonts.ready.then(() =>
  [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family.replace(/"/g, '') + ' ' + f.weight)
    .concat(['800 48px "Bricolage Grotesque"', '400 16px "IBM Plex Sans"', '600 16px "IBM Plex Sans"',
             '400 13px "IBM Plex Mono"', '600 13px "IBM Plex Mono"']
            .filter(f => !document.fonts.check(f)).map(f => 'not ready: ' + f)))"""
LOADED = ["Bricolage Grotesque 200 800", "IBM Plex Mono 400", "IBM Plex Mono 600",
          "IBM Plex Sans 400", "IBM Plex Sans 600"]


@pytest.mark.parametrize("seed", ["dark", "light"])
@pytest.mark.parametrize("page", sorted(BROWSER_PAGES))
def test_every_visible_text_meets_aa_in_every_state(open_page, page, seed):
    tab = open_page(page, seed=seed)
    assert tab.evaluate(THEME) == seed
    found = _demo_states(tab) if page == "demo" else _writeup_states(tab)
    assert not found, sorted(set(found))
    assert sorted(set(tab.evaluate(FACES))) == LOADED, "the self-hosted type did not all load"
    assert not tab.errors, tab.errors
