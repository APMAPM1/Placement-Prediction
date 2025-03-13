from flask import Flask, request, render_template
import joblib
import numpy as np
from http import HTTPStatus

app = Flask(__name__, template_folder='../templates')
model = joblib.load('../placement_model.pkl')
scaler = joblib.load('../scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(request.form[col]) for col in ['CGPA', 'Internships', 'Projects', 
                                                    'Workshops/Certifications', 'AptitudeTestScore', 
                                                    'SoftSkillsRating', 'ExtracurricularActivities', 
                                                    'PlacementTraining', 'SSC_Marks', 'HSC_Marks']]
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]
        result = 'Placed' if prediction == 1 else 'Not Placed'
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    except Exception as e:
        return str(e), HTTPStatus.BAD_REQUEST

# Vercel handler
def handler(request):
    from wsgi import wsgi_handler
    return wsgi_handler(app, request)
