import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. Load and prepare the data
# -----------------------------
fake_df = pd.read_csv("fake.csv")
true_df = pd.read_csv("true.csv")

# Assign labels to each dataset: 0 = fake, 1 = real
fake_df["label"] = 0
true_df["label"] = 1

# Combine and shuffle the dataset
data = pd.concat([fake_df, true_df], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["title"]
y = data["label"]

# -----------------------------
# 2. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Text vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# -----------------------------
# 4. Train the classifier
# -----------------------------
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vectors, y_train)

# -----------------------------
# 5. Evaluate the model
# -----------------------------
predictions = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# -----------------------------
# 6. Save model + vectorizer
# -----------------------------
joblib.dump(model, "model/fake_news_model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")

print("Training complete. Model and vectorizer saved successfully.")
