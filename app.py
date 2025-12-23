import gradio as gr
import joblib
import numpy as np
import requests

# ----------------------------------------------------
# Load the trained model and vectorizer
# ----------------------------------------------------
model = joblib.load("model/fake_news_model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# Your NewsAPI key
NEWS_API_KEY = "ENter Your Key"

# ----------------------------------------------------
# Primary ML prediction with confidence
# ----------------------------------------------------
def ml_predict(text, threshold=0.6):
    try:
        X = vectorizer.transform([text])
        score = model.decision_function(X)[0]
        confidence = abs(score)

        prediction = model.predict(X)[0]  # 1 = Real, 0 = Fake

        # Trigger fallback if confidence is low
        if confidence < threshold:
            return "uncertain", confidence

        label = "real" if prediction == 1 else "fake"
        return label, confidence

    except:
        return "uncertain", 0.0

# ----------------------------------------------------
# Secondary fallback: NewsAPI check
# ----------------------------------------------------
def newsapi_check(query):
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en"
        res = requests.get(url)
        data = res.json()

        if data.get("totalResults", 0) == 0:
            return "No news articles found."

        article = data["articles"][0]
        title = article.get("title", "No title available")
        description = article.get("description", "No description available")

        return f"Title: {title}\n\nDescription: {description}"

    except:
        return "NewsAPI lookup failed."

# ----------------------------------------------------
# Final hybrid decision
# ----------------------------------------------------
def final_output(text):
    ml_result, conf = ml_predict(text)

    # High confidence → trust ML
    if ml_result != "uncertain":
        output_label = "REAL NEWS ✓" if ml_result == "real" else "FAKE NEWS ✗"
        return f"{output_label}\nConfidence: {conf:.2f}"

    # Low confidence → fallback to NewsAPI
    news_result = newsapi_check(text)

    if news_result != "No news articles found.":
        return f"UNCERTAIN → Using NewsAPI:\n\n{news_result}"

    return "UNCERTAIN → No reliable information found."

# ----------------------------------------------------
# Gradio interface
# ----------------------------------------------------
demo = gr.Interface(
    fn=final_output,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news headline..."),
    outputs="text",
    title="Hybrid Fake News Detector (ML + NewsAPI)",
    description="Machine learning predicts first; if confident, returns a label. If not, the system checks live news articles using NewsAPI.",
    allow_flagging="never"
)

demo.launch()
