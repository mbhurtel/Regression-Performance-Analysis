#Decision Tree
from sklearn.tree import DecisionTreeRegressor

#Ensembes Models -> RandomForest 
from sklearn.ensemble import RandomForestRegressor

#KNN
from sklearn.neighbors import KNeighborsRegressor

# Linear Regression
from sklearn.linear_model import LinearRegression

#Support Vector Machine
from sklearn.svm import SVR

#ANN
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np

def train_data(X_train, X_test, y_train, y_test, prompt):
  
    if prompt == "RF":
        model = RandomForestRegressor(max_depth=20)
        regressor = "Random Forest"
    
    elif prompt == "KNN":
        model = KNeighborsRegressor(n_neighbors=5)
        regressor = "KNN"
        
    elif prompt == "DT":
        model = DecisionTreeRegressor(max_depth=10)
        regressor = "Decision Tree"
    
    elif prompt == "LR":
        model = LinearRegression()
        regressor = "Linear Regression"
        
    elif prompt == "LSVR":
        model = SVR(kernel='linear')
        regressor = "Linear SVR"
    
    elif prompt == "RBFSVR":
        model = SVR(kernel='rbf')
        regressor = "RBF Kernel SVR"
    
    elif prompt == "PSVR":
        model = SVR(kernel='poly')
        regressor = "Polynomial SVR"
    
    elif prompt == "ANN":
        regressor = "ANN"
        model = Sequential()
        model.add(Dense(input_dim=4, units=6, activation='tanh'))
        model.add(Dense(units=4, activation='tanh'))
        model.add(Dense(units=4, activation='tanh'))
        model.add(Dense(units=3, activation='relu'))
        model.compile(loss="mse", metrics=['mae'], optimizer='adam')

    else:
      print("Please enter a valid regression model!")
      assert False

    print(f"\nWORKING FOR {regressor.upper()} MODEL")

    # optimal_r2 = 0    
    coffs, intercepts = None, None
    
    if "SVR" in prompt:
        y_pred = []
        r2s = []
        coffs = []
        intercepts = []
        mapping = {0:"a", 1:"b", 2:"c"}
        
        for i in mapping:
            model.fit(X_train, y_train[:, i])
            y_pred.append(model.predict(X_test))
            r2s.append(round(r2_score(y_test[:, i], model.predict(X_test)), 3))
            if prompt == "LSVR":
              coffs.append(np.round(model.coef_, 3))
            else:
              coffs.append(np.round(model.dual_coef_, 3))
            intercepts.append(np.round(model.intercept_[0], 3))
            
        y_pred = np.array(y_pred).T
        result = round(np.array(r2s).mean(), 3)
        print(f"R2 Score: {result}")
            
    else:                
        if prompt == "ANN":
            model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=0)
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        result = round(r2_score(y_test, y_pred), 3)
        print(f"R2 Score: {result}")
        
    return y_pred, regressor, model, result, coffs, intercepts