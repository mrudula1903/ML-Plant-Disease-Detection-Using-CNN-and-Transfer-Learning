from flask import Flask, render_template, request
import pandas as pd
import webbrowser
import joblib

app = Flask(__name__)

model = joblib.load("plant_disease_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        data = {
            "Plant_Name": request.form["plant"],
            "Temperature_C": float(request.form["temperature"]),
            "Humidity_Percent": float(request.form["humidity"]),
            "Rainfall_mm": float(request.form["rainfall"]),
            "Soil_pH": float(request.form["ph"]),
            "Nitrogen": float(request.form["nitrogen"]),
            "Phosphorus": float(request.form["phosphorus"]),
            "Potassium": float(request.form["potassium"]),
            "Leaf_Spot_Size_cm2": float(request.form["leaf_spot"]),
            "Leaf_Color": request.form["leaf_color"]
        }

        input_df = pd.DataFrame([data])
        processed_data = preprocessor.transform(input_df)
        result = model.predict(processed_data)
        prediction = label_encoder.inverse_transform(result)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
