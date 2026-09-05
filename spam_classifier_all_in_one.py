"""
AI Spam Message Classifier — All-in-One Script
================================================
A single self-contained file that combines everything from the project:
the labeled dataset, preprocessing, model training/evaluation, model
saving, and a command-line interface for classifying new messages.

WHAT IT DOES
------------
1. Loads a built-in labeled dataset of spam/ham text messages
2. Cleans the text (lowercase, strip URLs/digits/punctuation)
3. Converts text to numeric features with TF-IDF (unigrams + bigrams)
4. Trains and compares two models: Multinomial Naive Bayes and
   Logistic Regression, keeping whichever scores higher on F1
5. Prints accuracy, precision, recall, F1, and a confusion matrix
6. Saves the trained model + vectorizer to disk (spam_model.joblib,
   vectorizer.joblib) so they can be reused without retraining
7. Lets you classify new messages, either via command-line arguments,
   an interactive prompt, or by importing predict_message() elsewhere

USAGE
-----
    python spam_classifier_all_in_one.py                # train + demo
    python spam_classifier_all_in_one.py --classify "Free prize! Click now"
    python spam_classifier_all_in_one.py --interactive   # type messages, Ctrl+C to quit

EXTENDING THIS PROJECT
-----------------------
- More data: replace DATASET_CSV below (or point load_data at an external
  CSV) with a larger real-world dataset, keeping the "label,text" format.
- Try other models: sklearn.svm.LinearSVC or
  sklearn.neural_network.MLPClassifier are easy drop-in swaps.
- Deploy it: wrap predict_message() in a Flask/FastAPI endpoint to filter
  messages in real time.
"""

import io
import re
import string
import argparse

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Resolved against this file, not the working directory. They used to be
# bare relative names, which meant the artifacts landed wherever you happened
# to be standing when you ran the script -- and app.py compensated by calling
# os.chdir() at import time, a process-wide side effect of an import.
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(_HERE, "vectorizer.joblib")

# The split, named once and imported by anything that needs to reproduce it.
#
# app.evaluate() scores a model loaded from disk by rebuilding this exact
# split and testing on the held-out quarter. That is only held out if it is
# the *same* split the model was trained with -- both numbers used to be
# written out separately in both files, so changing the split here and not
# there would have had the app quietly scoring a model against its own
# training data and reporting the inflated number as accuracy.
TEST_SIZE = 0.25
RANDOM_STATE = 42

# TF-IDF over unigrams and bigrams: "free" and "call now" both carry signal,
# and bigrams are what let the explanation name a phrase rather than a word.
NGRAM_RANGE = (1, 2)
MIN_DF = 1

# ---------------------------------------------------------------------------
# Embedded dataset (originally dataset.csv) — 81 labeled sample messages
# ---------------------------------------------------------------------------
DATASET_CSV = """label,text
spam,"Congratulations! You've won a $1000 Walmart gift card. Click here to claim now"
spam,"URGENT! Your account has been suspended. Verify your details immediately at this link"
spam,"You have been selected for a FREE cruise to the Bahamas! Call now to claim"
spam,"WINNER! As a valued customer you have been selected to receive a prize. Reply YES"
spam,"Limited time offer! Get 90% off on all products. Click the link before it expires"
spam,"Your loan of $5000 has been approved. No credit check required. Apply now"
spam,"Free entry into our $250 weekly draw, text WIN to 80086 now"
spam,"Claim your free iPhone 15 today! Just pay shipping and handling"
spam,"Hot singles in your area are waiting to chat with you tonight"
spam,"You have 1 new voicemail from an unknown caller, dial this premium number to listen"
spam,"CONGRATULATIONS your mobile number has won 500000 in the UK National Lottery"
spam,"Act now! Your credit card has been charged $499, call this number to dispute"
spam,"Get rich quick with this one simple trick banks dont want you to know"
spam,"Your package could not be delivered, click here to reschedule and pay a small fee"
spam,"FREE VIAGRA sample pack, no prescription needed, order online today"
spam,"You are pre approved for a $10000 personal loan, click to accept instantly"
spam,"Text STOP to unsubscribe or continue to receive amazing deals every day"
spam,"Your Netflix subscription payment failed, update your billing info immediately at this link"
spam,"Earn $5000 a week working from home, no experience necessary, sign up now"
spam,"This is your final notice, your car warranty is about to expire, call now"
spam,"Bitcoin investment opportunity, double your money in 24 hours guaranteed"
spam,"You've been chosen for a exclusive investment opportunity with 300% returns"
spam,"Click here to see who viewed your profile, limited time free access"
spam,"IRS notice: you owe back taxes, pay immediately to avoid arrest, click link"
spam,"Your Amazon order has an issue, verify your payment info within 24 hours or account will close"
spam,"Free trial ends today, enter your card details now to avoid interruption"
spam,"Congratulations you have been randomly selected to win a brand new laptop"
spam,"Reply now to claim your reward points before they expire tonight"
spam,"Special promo just for you, buy one get one free on all electronics today only"
spam,"Your bank account has unusual activity, confirm your identity now by clicking below"
spam,"Make money fast! Join our team and earn passive income with zero effort"
spam,"You qualify for a government grant of $9000, apply before the deadline"
spam,"Last chance to lower your mortgage rate, call this toll free number now"
spam,"Your subscription box is ready to ship, confirm your address to avoid cancellation fee"
spam,"Exclusive deal: get a free vacation package, just complete this short survey"
spam,"Your PayPal account has been limited, click here to restore full access"
spam,"You have an unclaimed inheritance of $2 million waiting, contact us to process"
spam,"Double your crypto in a week, join thousands of investors making profit daily"
spam,"Hurry the sale ends in 1 hour, use code SAVE70 for massive discounts"
spam,"Your number has won a lucky draw prize of an all expense paid trip"
ham,"Hey are we still on for lunch tomorrow at noon"
ham,"Can you pick up some milk on your way home"
ham,"The meeting has been moved to 3pm, see you in the conference room"
ham,"Happy birthday! Hope you have an amazing day"
ham,"I finished the report, let me know if you want any changes"
ham,"Running a bit late, be there in 10 minutes"
ham,"Thanks for helping me move this weekend, really appreciate it"
ham,"Did you watch the game last night, what a finish"
ham,"Mom said dinner is ready, come downstairs"
ham,"Let's catch up this weekend, it's been too long"
ham,"I attached the notes from today's class, let me know if you have questions"
ham,"Can you send me the address for the party on Saturday"
ham,"The doctor's appointment got rescheduled to next Tuesday at 2pm"
ham,"Great job on the presentation today, the client loved it"
ham,"Don't forget to bring your laptop charger tomorrow"
ham,"I'll be at the airport around 6, can you pick me up"
ham,"The kids have soccer practice at 5, can you drop them off"
ham,"Just wanted to check in and see how you're doing"
ham,"Can we reschedule our call to Thursday instead of Wednesday"
ham,"I left the keys under the mat, let yourself in"
ham,"Thanks for the recommendation, the restaurant was fantastic"
ham,"Are you free to talk tonight, I need some advice"
ham,"The project deadline got extended to next Friday"
ham,"I'm heading to the gym, want to join me later"
ham,"Congrats on the new job, you totally deserve it"
ham,"Can you review my essay before I submit it tomorrow"
ham,"The wifi is down again, can you call the provider"
ham,"Let's plan a trip for spring break, any ideas where to go"
ham,"I picked up the dry cleaning, it's hanging in the closet"
ham,"Your flight got delayed by two hours, new departure is 8pm"
ham,"Remember to water the plants while I'm away this week"
ham,"The book club meeting is at my place this time"
ham,"I updated the spreadsheet with the latest numbers"
ham,"Can you grab the mail when you get home"
ham,"We should carpool to the conference next month"
ham,"I made your favorite dinner, come eat before it gets cold"
ham,"The car needs an oil change, I'll take it in this weekend"
ham,"Thanks for covering my shift yesterday, I owe you one"
ham,"Let me know when you land so I can come pick you up"
ham,"The printer is out of toner again, did you order more"
ham,"I scheduled our dentist appointments for the same day next week"
"""


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation noise from a message."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"\d+", " ", text)                        # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def load_data() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(DATASET_CSV))
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df["clean_text"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF,
                                 stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    best_model, best_score, best_name = None, -1, None

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 score : {f1:.3f}")
        print("Confusion matrix [[TN FP]\n                  [FN TP]]:")
        print(confusion_matrix(y_test, preds))

        if f1 > best_score:
            best_score, best_model, best_name = f1, model, name

    print("\n" + "=" * 60)
    print(f"Best model: {best_name} (F1 = {best_score:.3f})")
    print("=" * 60)
    print("\nDetailed report for best model:")
    print(classification_report(
        y_test, best_model.predict(X_test_vec), target_names=["ham", "spam"]
    ))

    return best_model, vectorizer


def save_model(model, vectorizer):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved trained model to '{MODEL_PATH}'")
    print(f"Saved vectorizer to '{VECTORIZER_PATH}'")


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_message(message: str, model, vectorizer) -> str:
    """Classify a single new message as 'spam' or 'ham'."""
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    # Indexed by the model's class order, not by the label value. They
    # coincide here because the labels are mapped to 0/1 before training, so
    # classes_ is [0, 1] -- change that mapping and the old version would
    # quietly report the wrong class's probability as the confidence.
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(vec)[0][list(model.classes_).index(pred)])
    label = "spam" if pred == 1 else "ham"
    confidence = f" (confidence: {prob:.2%})" if prob is not None else ""
    return f"{label}{confidence}"


def run_training_and_demo():
    print("Loading dataset...")
    df = load_data()
    print(f"Loaded {len(df)} messages "
          f"({(df['label'] == 'spam').sum()} spam, {(df['label'] == 'ham').sum()} ham)\n")

    model, vectorizer = train_and_evaluate(df)
    save_model(model, vectorizer)

    print("\n" + "=" * 60)
    print("TRY IT OUT: classifying new sample messages")
    print("=" * 60)
    sample_messages = [
        "Congratulations, you have won a free ticket! Click here to claim now",
        "Hey, are you free for dinner tonight?",
        "URGENT: verify your bank account now or it will be suspended",
        "Can you send me the meeting notes from earlier?",
    ]
    for msg in sample_messages:
        result = predict_message(msg, model, vectorizer)
        print(f"  \"{msg}\"\n   --> {result}\n")


def run_interactive(model, vectorizer):
    print("Spam classifier — type a message and press Enter (Ctrl+C to quit)\n")
    try:
        while True:
            message = input("> ")
            if message.strip():
                print(" ", predict_message(message, model, vectorizer))
    except KeyboardInterrupt:
        print("\nBye!")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI Spam Message Classifier (all-in-one)")
    parser.add_argument("--classify", type=str, default=None,
                        help="Classify a single message and exit")
    parser.add_argument("--interactive", action="store_true",
                        help="Classify messages interactively (Ctrl+C to quit)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if a saved model already exists")
    args = parser.parse_args()

    if args.classify or args.interactive:
        model, vectorizer = load_artifacts()
        if model is None or args.retrain:
            print("No saved model found (or --retrain used) — training first...\n")
            df = load_data()
            model, vectorizer = train_and_evaluate(df)
            save_model(model, vectorizer)
            print()

        if args.classify:
            print(predict_message(args.classify, model, vectorizer))
        else:
            run_interactive(model, vectorizer)
        return

    # Default: train, evaluate, save, and demo
    run_training_and_demo()


if __name__ == "__main__":
    main()
