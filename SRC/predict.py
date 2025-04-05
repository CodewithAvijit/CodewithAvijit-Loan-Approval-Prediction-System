import pandas as pd
import numpy as np
import joblib

model = joblib.load('DATA/model.pkl')

education = input("ENTER YOUR EDUCATION: ").strip().lower()
selfemployment = input("ARE YOU SELF EMPLOYED: ").strip().lower()
income = float(input("ENTER YOUR ANNUAL INCOME: "))
loan = float(input("ENTER YOUR LOAN AMOUNT: "))
civil_score = float(input("ENTER YOUR CIVIL SCORE: "))
residential_assets_value = float(input("ENTER YOUR RESIDENTIAL ASSETS VALUE: "))
commercial_assets_value = float(input("ENTER YOUR COMMERCIAL ASSETS VALUE: "))

if education == "graduate":
    education = 1
elif education == "not graduate":
    education = 0
else:
    print("INVALID EDUCATION")
    exit()

if selfemployment == "yes":
    self_employed = 1
elif selfemployment == "no":
    self_employed = 0
else:
    print("INVALID SELF EMPLOYED")
    exit()

input_data = np.array([[education, self_employed, income, loan, civil_score, residential_assets_value, commercial_assets_value]])

prediction = model.predict(input_data)[0]

print("LOAN STATUS:")
if prediction == 1:
    print("LOAN APPROVED")
else:
    print("LOAN REJECTED")
