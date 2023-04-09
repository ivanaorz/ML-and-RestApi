from flask import Flask, request, jsonify
import joblib

svm_model = joblib.load("heart-disease-persisted-model.joblib")

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

   
    # Getting the feature inputs from the user
    feature_inputs = request.get_json()

    # Extracting the feature values
    age = feature_inputs["age"]
    sex = feature_inputs["sex"]
    chest_pain_type = feature_inputs["chest_pain_type"]
    bp = feature_inputs["bp"]
    cholesterol = feature_inputs["cholesterol"]
    fbs_over_120 = feature_inputs["fbs_over_120"]
    ekg_results = feature_inputs["ekg_results"]
    max_hr = feature_inputs["max_hr"]
    exercise_angina = feature_inputs["exercise_angina"]
    st_depression = feature_inputs["st_depression"]
    slope_of_st = feature_inputs["slope_of_st"]
    number_of_vessels_fluro = feature_inputs["number_of_vessels_fluro"]
    thallium = feature_inputs["thallium"]

    # Creating a feature vector for the model
    feature_vector = [age, sex, chest_pain_type, bp, cholesterol, fbs_over_120,
                      ekg_results, max_hr, exercise_angina, st_depression, slope_of_st,
                      number_of_vessels_fluro, thallium]

    # Making prediction
    prediction = svm_model.predict([feature_vector])

   
   
    # Returning the prediction
    if prediction[0] == 1:
        result = 'has heart disease'
    else:
        result = "doesn't have heart disease"
    return jsonify({'prediction': result})
   

if __name__ == "__main__":
    app.run(debug=True)

    # Input data for testing in Postman

    '''
    "age": 35,
    "sex": 0,
    "chest_pain_type": 1,
    "bp": 180,
    "cholesterol": 250,
    "fbs_over_120": 1,
    "ekg_results": 3,
    "max_hr": 121,
    "exercise_angina": 0,
    "st_depression": 1.2,
    "slope_of_st": 1,
    "number_of_vessels_fluro": 3,
    "thallium": 7
    '''

    '''
    "age": 65,
    "sex": 0,
    "chest_pain_type": 1,
    "bp": 110,
    "cholesterol": 300,
    "fbs_over_120": 0,
    "ekg_results": 3,
    "max_hr": 160,
    "exercise_angina": 0,
    "st_depression": 1.2,
    "slope_of_st": 2,
    "number_of_vessels_fluro": 0,
    "thallium": 3
    '''


    '''
    "age": 100,
    "sex": 1,
    "chest_pain_type": 3,
    "bp": 180,
    "cholesterol": 500,
    "fbs_over_120": 0,
    "ekg_results": 2,
    "max_hr": 160,
    "exercise_angina": 0,
    "st_depression": 1.2,
    "slope_of_st": 2,
    "number_of_vessels_fluro": 0,
    "thallium": 7
    '''