from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Pages
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

# -------------------------------
# Prediction
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if prediction == 1:
      result = f"Spam ❌ ({prob*100:.2f}% confidence)"
      status = "spam"
    else:
      result = f"Not Spam ✅ ({(1-prob)*100:.2f}% confidence)"
      status = "ham"

    return render_template("detect.html", prediction_text=result, user_input=message, status=status)
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)