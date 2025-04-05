import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
def iqr(df,column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].dropna()
    
data=pd.read_csv(r"DATA/rawdata.csv")
print(data['education'])
# print(data.info())
# print(data.head())
# print(data.columns.tolist())
# sns.heatmap(data.isnull())
# plt.show()
lr=LabelEncoder()
data['education']=lr.fit_transform(data['education'])
data['self_employed']=lr.fit_transform(data['self_employed'])
data['loan_status']=lr.fit_transform(data['loan_status'])
mscol=['income_annum', 'loan_amount', 'civil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
ms=MinMaxScaler()
for col in mscol:
    if col in data.columns:
        data[col]=ms.fit_transform(data[[col]])
# sns.boxplot(x='commercial_assets_value',data=data)
# plt.show() # use to check outliers
for col in mscol:
    if col in data.columns:
        data = iqr(data,col)
# sns.boxplot(x='commercial_assets_value',data=data)
# plt.show()
# print(data)
data.to_csv(r"DATA/processeddata.csv",index=False)