from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('placement_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form[col]) for col in ['CGPA', 'Internships', 'Projects', 
                                                'Workshops/Certifications', 'AptitudeTestScore', 
                                                'SoftSkillsRating', 'ExtracurricularActivities', 
                                                'PlacementTraining', 'SSC_Marks', 'HSC_Marks']]
    data_scaled = scaler.transform([data])
    prediction = model.predict(data_scaled)[0]
    result = 'Placed' if prediction == 1 else 'Not Placed'
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)