"""#Importing Libraries"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from display import *
from helper import *

"""#Importing the regression models"""

# Function required at last
pad = lambda y_true, y_pred: ( np.abs(y_true - y_pred) / y_true) * 100

"""#Importing and Processing Data"""

columns = ["compound", "ra", "rb", "fe", "bg", "a", "b", "c", "shape"]
inputs = {0:"ra", 1:"rb", 2:"fe", 3:"bg"}
mapping = {0:"a", 1:"b", 2:"c"}

data = pd.read_excel("cubic and orthorhomic.xlsx", sheet_name='cubic')
data.columns = columns

#Separating the dependent and independent variables
X = data.iloc[:, 1:5].dropna().values
y = data.iloc[:, 5:-1].dropna().values

#Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = scaler.transform(X)

#Dictionaries to store the r2_scores from different regressors
r2_scores = {}
predictions = {}

"""#Assigining the regressor and working with it"""

'''
NOTE:
Random Forest = RF
Decision Tree = DT
Linear Regression = LR
K Nearest Neighbour = KNN
Linear SVR = LSVR
Polynomial SVR = PSVR
RBF Kernel SVR = RBFSVR
'''

r_models = ["RF", "DT", "LR", "KNN", "LSVR", "PSVR", "RBFSVR", "ANN"]

for r_model in r_models:
    #Prompt to get the regression model
    prompt = r_model

    #Getting the best 16-folds cross validated data
    X_train, X_test, y_train, y_test, y_pred, regressor, model, r2_score, coffs, intercepts = get_best_data(X,y, prompt)
    print(f"Optimal R2 Score for {regressor} is {r2_score}")

    #Storing the r2_score and predictions in the dictionaries
    r2_scores[regressor] = r2_score
    predictions[regressor] = y_pred

    # Analysis for Random Forest Regressor:
    if prompt == "RF":
        print("\nFeature Importance:")
        for i,c in enumerate(columns[1:-4]):
            print(f"{c}:\t{model.feature_importances_[i].round(3)}")

    # Analysis for Linear Regressor:
    if prompt == "LR":
        coffs = np.round(model.coef_, 3)
        intercepts = np.round(model.intercept_, 3)
        
        print("\nApproximated Equations from Linear Model:")
        for i in mapping:
            print(f"{mapping[i]} = {coffs[i][0]}{inputs[0]} + {coffs[i][1]}{inputs[1]} + {coffs[i][2]}{inputs[2]} + {coffs[i][3]}{inputs[3]} + {intercepts[i]}")

    # Analysis for Support Vector Regressor:
    if "SVR" in prompt:
        print(f"\nApproximated Equations from {regressor} Model: ")
        for i in mapping:
            print(f"{mapping[i]} = {coffs[i][0][0]}{inputs[0]} + {coffs[i][0][1]}{inputs[1]} + {coffs[i][0][2]}{inputs[2]} + {coffs[i][0][3]}{inputs[3]} + {intercepts[i]}")

    #Display Plots
    exp_vs_pred_subplots(y_test, y_pred, regressor)

    exp_vs_pred_lc(y_test, y_pred, regressor, r2_score)

    names = X_test[:, 0]
    exp_vs_pred_plotly(y_test, y_pred, regressor, names)

"""#Comparision of performance of all regression models"""

comparision(r2_scores)

"""#Calculating PAD and Predicted value for Table"""

prediction_table = pd.concat([
       pd.DataFrame(v, columns=['a', 'b', 'c'], index=np.repeat(k, len(v))) 
       for k, v in predictions.items()
  ]
).rename_axis('algorithm').reset_index()

print(prediction_table.head())

pads = {}
for algo, y_pred in predictions.items():
    pads[algo] = pad(y_test, y_pred)
    
pad_table = pd.concat([
       pd.DataFrame(v, columns=['a', 'b', 'c'], index=np.repeat(k, len(v))) 
       for k, v in pads.items()
  ]
).rename_axis('algorithm').reset_index()

print(pad_table.head())