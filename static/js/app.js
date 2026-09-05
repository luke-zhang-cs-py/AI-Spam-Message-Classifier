/*
 * app.js — page wiring for the spam classifier.
 *
 * All the work happens server-side; this posts text and renders the verdict,
 * the confidence, and the per-word contributions that produced it.
 */

const $ = (id) => document.getElementById(id);

const input = $('input');
const errBox = $('err');

const SAMPLES = [
  "Congratulations! You've won a $1000 gift card. Click here to claim now",
  'Hey are we still on for lunch tomorrow at noon',
  'URGENT: your account has been suspended, verify your details immediately',
  'Can you review my essay before I submit it tomorrow',
  'Free entry into our weekly draw, text WIN to 80086 now',
];

async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({ ok: false, error: 'Bad response' }));
  if (!res.ok || data.ok === false) throw new Error(data.error || 'Request failed');
  return data;
}

function showError(msg) {
  errBox.textContent = msg;
  errBox.style.display = msg ? 'block' : 'none';
}

const pct = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');
const esc = (s) => s.replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ------------------------------------------------------------- single */

function renderResult(r) {
  $('batchCard').style.display = 'none';
  $('resultCard').style.display = 'block';

  $('badge').textContent = r.label;
  $('badge').className = 'badge ' + r.label;
  $('confVal').textContent = pct(r.confidence);
  $('confFill').style.width = ((r.confidence || 0) * 100) + '%';
  $('confFill').className = 'fill ' + r.label;

  $('knownTokens').textContent = r.knownTokens;
  $('cleaned').textContent = r.cleaned || '(nothing left after cleaning)';

  /* Weight sign is the direction: positive pushed the score towards spam,
   * negative towards ham. Magnitude is how hard. */
  $('tokens').innerHTML = r.tokens.length
    ? r.tokens.map((t) => {
        const dir = t.weight >= 0 ? 'spam' : 'ham';
        return `<span class="tok ${dir}"><b>${esc(t.token)}</b>
          <i>${t.weight >= 0 ? '+' : ''}${t.weight.toFixed(3)}</i></span>`;
      }).join('')
    : '<span class="note">No recognised words — every term is outside the ' +
      'vocabulary the model was trained on.</span>';

  $('tokenNote').textContent = r.knownTokens === 0
    ? 'With no known words the verdict falls back to the class priors, so treat it as a guess.'
    : `${r.knownTokens} of this message's terms appear in the training vocabulary. ` +
      'Words the model never saw contribute nothing.';
}

async function classify() {
  showError('');
  const message = input.value.trim();
  if (!message) { showError('Type a message first.'); return; }

  $('classifyBtn').disabled = true;
  try {
    renderResult(await post('/api/classify', { message }));
  } catch (e) { showError(e.message); }
  $('classifyBtn').disabled = false;
}

/* -------------------------------------------------------------- batch */

async function batch() {
  showError('');
  const messages = input.value.split('\n').map((l) => l.trim()).filter(Boolean);
  if (!messages.length) { showError('Type at least one line.'); return; }

  $('batchBtn').disabled = true;
  try {
    const d = await post('/api/batch', { messages });
    $('resultCard').style.display = 'none';
    $('batchCard').style.display = 'block';
    $('bTotal').textContent = d.summary.total;
    $('bSplit').textContent = `${d.summary.spam} / ${d.summary.ham}`;
    $('batchTable').innerHTML = d.results.map((r) => `
      <tr>
        <td class="v ${r.label}">${r.label}</td>
        <td class="c">${pct(r.confidence)}</td>
        <td class="m">${esc(r.message)}</td>
      </tr>`).join('');
  } catch (e) { showError(e.message); }
  $('batchBtn').disabled = false;
}

/* -------------------------------------------------------------- model */

function renderModel(d) {
  $('mName').textContent = d.name || '—';
  $('mData').textContent = d.datasetSize ? `${d.datasetSize} messages` : '—';

  const m = d.metrics;
  $('mAcc').textContent = m ? m.accuracy : '—';
  $('mPrec').textContent = m ? m.precision : '—';
  $('mRec').textContent = m ? m.recall : '—';
  $('mF1').textContent = m ? m.f1 : '—';
  $('mNote').textContent = m
    ? `Scored on a held-out split of ${m.testSize} messages.`
    : 'Scores appear after a retrain — a model loaded from disk carries no metrics with it.';
}

async function loadModel() {
  try {
    renderModel(await (await fetch('/api/model')).json());
  } catch (_) { /* page still usable */ }
}

/* ------------------------------------------------------------- wiring */

$('classifyBtn').onclick = classify;
$('batchBtn').onclick = batch;
$('clearBtn').onclick = () => {
  input.value = '';
  $('resultCard').style.display = 'none';
  $('batchCard').style.display = 'none';
  showError('');
  input.focus();
};

$('retrainBtn').onclick = async () => {
  showError('');
  const btn = $('retrainBtn');
  btn.disabled = true;
  btn.textContent = 'Retraining…';
  try {
    renderModel(await post('/api/retrain'));
  } catch (e) { showError(e.message); }
  btn.textContent = 'Retrain';
  btn.disabled = false;
};

/* Ctrl/Cmd+Enter classifies, so you never have to reach for the mouse. */
input.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') classify();
});

$('samples').innerHTML = SAMPLES.map((s) => `<li>${esc(s)}</li>`).join('');
[...$('samples').children].forEach((li, i) => {
  li.onclick = () => { input.value = SAMPLES[i]; classify(); };
});

loadModel();
input.focus();
