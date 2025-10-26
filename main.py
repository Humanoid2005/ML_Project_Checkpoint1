import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgbm
import xgboost as xgb
from catboost import CatBoostRegressor

from Preprocesor import Preprocessor

preprocessor = Preprocessor()
preprocessor.preprocess()

X = preprocessor.X
y = preprocessor.y
X_test = preprocessor.X_test
test_ids = preprocessor.test_ids


model_name = input("Enter the model you want to use: ")


if(model_name.lower() == 'bayesian'):
    model = BayesianRidge(max_iter=5000)

elif(model_name.lower() == 'knn'):
    model = KNeighborsRegressor(
        n_neighbors=100,weights='distance'
    )

elif(model_name.lower() == 'xgboost'):
    model  = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        min_split_loss = 0.1,

        n_estimators=10000,
        learning_rate=0.01,
        max_depth=7,
        
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5, 
        reg_alpha=0.1,
        reg_lambda=1.0
    )

elif(model_name.lower() == 'catboost'):
    model = CatBoostRegressor(
                    random_seed=42, 
                    iterations=4000,
                    learning_rate=0.02, depth=6, verbose=0,
                    l2_leaf_reg=5,
                    early_stopping_rounds=100
                )

elif(model_name.lower() == 'lightgbm'):
    model = lgbm.LGBMRegressor(
        random_state=42, n_jobs=-1,
        learning_rate=0.03,         
        n_estimators=10000,        
        num_leaves=32,             
        max_depth=6,                
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5              
    )

# model = RandomForestRegressor(
#     n_estimators=500,      
#     max_depth=8,           
#     min_samples_split=5,    
#     min_samples_leaf=1,    
#     max_features=0.8,    
#     n_jobs=-1,            
#     random_state=42
# )

# model = LinearRegression()
# model = Lasso(alpha=5.0)

X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(x_test)

y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_actual,y_pred_actual))
print(f"RMSE: {rmse}")

Y_test = model.predict(X_test)
Y_test = np.expm1(Y_test)

output_df = pd.DataFrame({
    'Hospital_Id':test_ids,
    'Transport_Cost':Y_test
})

folder_name = "submissions"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

output_df.to_csv(f"./submissions/{model_name.lower()}_submission.csv",index=False)