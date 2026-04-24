import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("spam.csv", encoding='latin-1')

# Fix column names
df = df[['Label', 'Text']]
df.columns = ['label', 'message']

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

# -------------------------------
# 3. Feature Extraction (TF-IDF)
# -------------------------------
cv = TfidfVectorizer(stop_words='english', max_features=5000)

X = cv.fit_transform(df['message'])
y = df['label']

# -------------------------------
# 4. Train-Test Split (WITH TEXT)
# -------------------------------
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, df['message'], test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train Model
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# 👉 Probability predictions
y_prob = model.predict_proba(X_test)

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
print("\n📊 MODEL PERFORMANCE\n")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. Show Sample Predictions
# -------------------------------
print("\n🔍 Sample Predictions with Confidence:\n")

for i in range(5):
    print(f"Message: {text_test.iloc[i][:80]}...")
    print(f"Prediction: {'Spam ❌' if y_pred[i]==1 else 'Ham ✅'}")
    print(f"Confidence (Spam %): {y_prob[i][1]*100:.2f}%\n")

# -------------------------------
# 9. Confusion Matrix Visualization
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 10. Save Model & Vectorizer
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("\n✅ Model and vectorizer saved successfully!")