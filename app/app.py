from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load(r'C:\Users\Avijit\Desktop\Loan Approval Prediction System\DATA\model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        education = request.form['education'].lower()
        selfemployment = request.form['selfemployment'].lower()
        income = float(request.form['income'])
        loan = float(request.form['loan'])
        civil_score = float(request.form['civil_score'])
        residential_assets_value = float(request.form['residential_assets_value'])
        commercial_assets_value = float(request.form['commercial_assets_value'])

        # Convert education and self-employment to binary values
        if education == "graduate":
            education = 1
        elif education == "notgraduate":
            education = 0
        else:
            education = 0

        if selfemployment == "yes":
            self_employed = 1
        elif selfemployment == "no":
            self_employed = 0
        else:
            return "Invalid self employment value"

        input_data = np.array([[education, self_employed, income, loan, civil_score, residential_assets_value, commercial_assets_value]])
        prediction = model.predict(input_data)[0]

        result = "LOAN APPROVED" if prediction == 1 else "LOAN REJECTED"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
