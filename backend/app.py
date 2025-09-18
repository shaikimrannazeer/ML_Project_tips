from flask import Flask, request, jsonify
# Flask is to create an api end point
# requet is to get the data from the streamlit app
# jsonify is to convert the data into json format and return it to the streamlit app
import joblib
import pandas as pd

app = Flask(__name__) # creating an object of the Flask class

# Load the trained model
model = joblib.load("random_forest_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to get predictions from the trained model."""

    data = request.json # Get data from the request {"total_bill": 16.99, "sex":"Male"....}


    input_df = pd.DataFrame([{"total_bill": data["total_bill"],
                            "sex": data["sex"],
                            "smoker": data["smoker"],
                            "day": data["day"],
                            "time": data["time"],
                            "size": data["size"]}])  # Convert data to DataFrame
     # Convert data to DataFrame
    prediction = model.predict(input_df)  # Make prediction # array[tip_value]
    return jsonify({"prediction": prediction[0]})  # Return prediction as JSON


if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app
