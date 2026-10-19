# Train the spam/ham model and save the .pkl files
# Put SpamSmsDataset.xlsx in the same folder before running this script.

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load the dataset
df = pd.read_excel("SpamSmsDataset.xlsx")

# 2. Keep rows with a message and label
df = df.dropna(subset=["message", "label"]).copy()
df["message"] = df["message"].astype(str)
df["label"] = df["label"].astype(str).str.strip().str.lower()

print("Dataset labels:")
print(df["label"].value_counts())

# 3. Convert labels into numbers
label_encoder = LabelEncoder()
df["label_num"] = label_encoder.fit_transform(df["label"])

# 4. Split messages and labels
X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Convert message text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("\nModel evaluation:")
print(classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
))

# 8. Save model files for Streamlit
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nSaved these files:")
print("- model.pkl")
print("- vectorizer.pkl")
print("- label_encoder.pkl")
print("\nUpload all three .pkl files to the same GitHub repository as app.py.")
