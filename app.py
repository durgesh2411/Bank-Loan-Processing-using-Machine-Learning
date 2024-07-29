import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("C:/Users/VISHU/3D Objects/Flask/Bank Loan Processing/Bank.pkl", "rb"))

# The HTML file should be created in Templates folder as a subdirectory...
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    fr = [float(x) for x in request.form.values()]
    features = [np.array(fr)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Accurate Loan Amount is Rs. {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)