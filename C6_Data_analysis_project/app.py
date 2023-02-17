from flask import Flask, render_template, request
from models import SentimentClassifier

classifier = SentimentClassifier()

app = Flask(__name__)


@app.route("/")
def man():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def home():
    text = request.form["text"]
    pred = classifier.get_prediction_message(text)
    return render_template("predict.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)
