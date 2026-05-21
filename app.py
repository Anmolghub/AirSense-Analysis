import os
import sqlite3
import pickle
import re
import nltk
import random
import string
from datetime import datetime
import pytz
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from database import init_db, DB_PATH

# ---------- NLTK SETUP ----------
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "airline_sentiment_secret_key")
init_db()

# ---------- LOAD MODEL ----------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------- HELPERS ----------
def get_ist_time():
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist).strftime('%Y-%m-%d %I:%M %p')

def generate_pnr():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def extract_keywords(text, top_n=5):
    return list(dict.fromkeys(preprocess_text(text).split()))[:top_n]

def airline_action(sentiment):
    if sentiment == "Negative":
        return "⚠️ CRITICAL: Route to Service Recovery Core. Dispatch tier-1 compensation options, issue systemic apology, and tag reservation for priority ground-handling support."
    elif sentiment == "Neutral":
        return "🟡 MONITOR: Log to continuous operational intelligence feed. Queue for routine aggregate text summarization; no immediate passenger-level outreach required."
    return "✅ ENHANCE: Forward to Marketing & Brand Advocacy. Flag for frequent flyer program (FFP) loyalty upgrade consideration and queue for automated social-sharing invitation."

CATEGORY_KEYWORDS = {
    "Delay & Disruption": ["delay", "late", "cancel", "waiting", "postpone", "schedule", "missed", "connection", "stranded", "overnight", "diverted"],
    "Staff & Service": ["service", "staff", "crew", "support", "attendant", "pilot", "agent", "helpdesk", "rude", "friendly", "hospitality", "assistance"],
    "Cabin Comfort & Dining": ["seat", "food", "comfort", "legroom", "cabin", "meal", "beverage", "cleanliness", "screen", "wifi", "entertainment", "recline"],
    "Baggage & Cargo": ["luggage", "bag", "baggage", "lost", "damaged", "carousel", "checkin", "weight", "allowance", "missing"],
    "Pricing & Ticketing": ["price", "cost", "expensive", "cheap", "fare", "refund", "fee", "charge", "hidden", "booking", "upgrade", "ticket"]
}

def detect_category(text):
    text = text.lower()
    for cat, words in CATEGORY_KEYWORDS.items():
        if any(w in text for w in words):
            return cat
    return "General Operations"

def predict_sentiment(text):
    clean = preprocess_text(text)
    vector = vectorizer.transform([clean])
    sentiment = model.predict(vector)[0]
    prob = model.predict_proba(vector).max()
    return sentiment, round(prob * 100, 2), extract_keywords(text)

# ---------- ROUTES ----------

@app.route("/", methods=["GET", "POST"])
def index():
    if "current_pnr" not in session:
        session["current_pnr"] = generate_pnr()

    session.setdefault("history", [])
    prediction = confidence = keywords = category = action = None
    review_text = ""
    error = None

    if request.method == "POST":
        review_text = request.form.get("review_text")
        pnr = session["current_pnr"]
        ist_now = get_ist_time()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT pnr FROM reviews WHERE pnr = ?", (pnr,))
        if cursor.fetchone():
            error = f"Review already submitted for PNR: {pnr}. Please 'Next Passenger' to continue."
            conn.close()
        elif len(review_text.strip()) < 5:
            error = "Review text is too short. Please provide a more descriptive summary of the flight experience to run analytics."
            conn.close()
        else:
            prediction, confidence, keywords = predict_sentiment(review_text)
            category = detect_category(review_text)
            action = airline_action(prediction)

            cursor.execute(
                "INSERT INTO reviews (pnr, review, sentiment, confidence, category, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (pnr, review_text, prediction, confidence, category, ist_now)
            )
            conn.commit()
            conn.close()

            session["history"].append({"pnr": pnr, "text": review_text, "sentiment": prediction, "time": ist_now})
            session.modified = True

    return render_template("index.html", prediction=prediction, confidence=confidence,
                           keywords=keywords, category=category, action=action,
                           review_text=review_text, history=session["history"],
                           current_pnr=session["current_pnr"], error=error)

@app.route("/next-passenger")
def next_passenger():
    session["current_pnr"] = generate_pnr()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(confidence) FROM reviews")
    stats = cursor.fetchone()
    total_reviews = stats[0] or 0
    avg_confidence = round(stats[1] or 0, 2)

    cursor.execute("SELECT sentiment, COUNT(*) FROM reviews GROUP BY sentiment")
    sentiment_counts = dict(cursor.fetchall())

    cursor.execute("SELECT category, COUNT(*) FROM reviews GROUP BY category")
    cat_data = dict(cursor.fetchall())

    cursor.execute(
        "SELECT category, review, sentiment, confidence, timestamp, pnr FROM reviews ORDER BY id DESC")
    full_logs = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", total_reviews=total_reviews, avg_confidence=avg_confidence,
                           sentiment_counts=sentiment_counts, cat_labels=list(cat_data.keys()),
                           cat_values=list(cat_data.values()), full_logs=full_logs)

@app.route("/download-csv")
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT pnr, review, sentiment, confidence, category, timestamp FROM reviews")
    rows = cursor.fetchall()
    conn.close()

    def generate():
        yield "PNR,Review,Sentiment,Confidence,Category,Timestamp_IST\n"
        for r in rows:
            review_clean = r[1].replace('"', '""')
            yield f"\"{r[0]}\",\"{review_clean}\",{r[2]},{r[3]},{r[4]},{r[5]}\n"

    return Response(generate(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=airline_analysis_ist.csv"})

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)