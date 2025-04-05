import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
data=pd.read_csv("DATA\processeddata.csv")
print(data.head())
x=data.iloc[:,:-1]
y=data['loan_status']
lr=LogisticRegression()
sf=SequentialFeatureSelector(lr,k_features='best',forward=True,scoring='accuracy')
us=RandomUnderSampler()
sf.fit(x,y)
x=sf.transform(x)
x,y=us.fit_resample(x,y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
lr.fit(xtrain,ytrain)

print(sf.k_feature_names_)
print(lr.score(xtrain,ytrain))
print(accuracy_score(ytest,lr.predict(xtest)))
print(confusion_matrix(ytest,lr.predict(xtest)))
print(classification_report(ytest,lr.predict(xtest)))
joblib.dump(lr,'DATA/model.pkl')