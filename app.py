from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#Load Model 
with open("decision_tree_model.pkl","rb")as f:
    model = pickle.load(f)

    species = {
        0:"Setosa",
        1:"Versicolor",
        2:"Virginica"
    }

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        data = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(data)[0]

        result = species[prediction]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Species:{result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)




















