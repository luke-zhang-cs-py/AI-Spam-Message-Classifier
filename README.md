# AI Spam Message Classifier

A small, self-contained machine learning project that classifies text
messages as **spam** or **ham** (not spam) using scikit-learn.

## Files

| File | Purpose |
|---|---|
| `dataset.csv` | 81 labeled sample messages (spam/ham) used for training |
| `train_spam_classifier.py` | Cleans data, extracts TF-IDF features, trains & compares two models (Naive Bayes, Logistic Regression), prints metrics, saves the best model |
| `classify.py` | Command-line tool to classify new messages using the saved model |
| `spam_model.joblib` / `vectorizer.joblib` | Saved trained model + vectorizer (created after training) |

## How it works

1. **Preprocessing** — lowercase text, strip URLs/digits/punctuation.
2. **Feature extraction** — `TfidfVectorizer` with unigrams + bigrams turns
   each message into a numeric vector, down-weighting common English words.
3. **Model** — `MultinomialNB` (Naive Bayes) is compared against
   `LogisticRegression`; whichever scores higher on F1 is kept.
4. **Evaluation** — accuracy, precision, recall, F1, and a confusion matrix
   on a held-out test split.
5. **Inference** — `predict_message()` / `classify.py` clean and vectorize
   a new message the same way, then run it through the saved model.

## Usage

Train the model (also runs evaluation and a few example predictions):

```bash
python train_spam_classifier.py
```

Classify a message from the command line:

```bash
python classify.py "Congratulations! You've won a free prize, click here"
```

Or run it interactively:

```bash
python classify.py
> Hey, want to grab lunch tomorrow?
  HAM (68.2% confidence)
```

## Extending this project

- **More data**: swap `dataset.csv` for a larger real-world dataset (e.g. the
  classic SMS Spam Collection) for better accuracy — just keep the same
  `label,text` column format.
- **Try other models**: SVM (`sklearn.svm.LinearSVC`) or a simple neural net
  (`sklearn.neural_network.MLPClassifier`) are easy drop-in swaps.
- **Deploy it**: wrap `predict_message()` in a Flask/FastAPI endpoint to
  filter messages in real time.
