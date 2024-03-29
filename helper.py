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

def get_best_data(X, y, prompt):
    
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
    kf = KFold(n_splits = 16)

    optimal_r2 = 0    
    coffs1, intercepts1 = None, None
    
    if "SVR" in prompt:
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                                                 y[train_index], y[test_index]
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
            
            if result >= optimal_r2:            
                optimal_r2 = result
                X_train1, X_test1, y_train1, y_test1, y_pred1, coffs1, intercepts1 = X_train,\
                                                X_test, y_train, y_test, y_pred, coffs, intercepts
                                 
    else:    
        for train_index, test_index in kf.split(X):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], \
                                                 y[train_index], y[test_index]
            
            if prompt == "ANN":
                model.fit(X_train, y_train, validation_split=0.1, epochs=100, verbose=0)
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict(X_test)
            result = round(r2_score(y_test, y_pred), 3)
            print(f"R2 Score: {result}")
            
            if result >= optimal_r2:
                optimal_r2 = result
                X_train1, X_test1, y_train1, y_test1, y_pred1 = X_train, X_test, \
                                                        y_train, y_test, y_pred
        
    return X_train1, X_test1, y_train1, y_test1, y_pred1, regressor, model, optimal_r2, coffs1, intercepts1