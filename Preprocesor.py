import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

class Preprocessor:
    def __init__(self):
        pass
    
    def target_encode_cv(self,train_series, y_series, test_series, n_splits=10, smoothing=20, seed=42):
        """KFold target encoding with smoothing (returns oof_train, test_encoded)."""
        oof = pd.Series(index=train_series.index, dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        global_mean = y_series.mean()
        for tr_idx, val_idx in kf.split(train_series):
            tr_s = train_series.iloc[tr_idx]
            tr_y = y_series.iloc[tr_idx]
            stats = pd.DataFrame({'cat': tr_s, 'y': tr_y}).groupby('cat')['y'].agg(['mean','count'])
            counts, means = stats['count'], stats['mean']
            smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
            mapping = smooth.to_dict()
            oof.iloc[val_idx] = train_series.iloc[val_idx].map(mapping).fillna(global_mean)
            
        full_stats = pd.DataFrame({'cat': train_series, 'y': y_series}).groupby('cat')['y'].agg(['mean','count'])
        counts, means = full_stats['count'], full_stats['mean']
        smooth_full = (counts * means + smoothing * global_mean) / (counts + smoothing)
        test_mapped = test_series.map(smooth_full.to_dict()).fillna(global_mean)
        return oof, test_mapped
    
    def preprocess(self,train_csv="train.csv",test_csv="test.csv"):
        
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        target_column = "Transport_Cost"
        self.test_ids = test_df['Hospital_Id']
        
        # 1. Fill NaN values
        
        train_df = train_df.drop_duplicates()
        
        string_columns = ["Hospital_Id","Supplier_Name","Equipment_Type","CrossBorder_Shipping","Urgent_Shipping",
                          "Installation_Service","Transport_Method","Fragile_Equipment","Hospital_Info","Rural_Hospital",
                          "Order_Placed_Date","Delivery_Date","Hospital_Location"]
        
        for column in string_columns:
            train_df[column] = train_df[column].astype(str).str.strip().str.lower()
            test_df[column] = test_df[column].astype(str).str.strip().str.lower()
        
        
        null_cat_columns = ["Equipment_Type","Transport_Method","Rural_Hospital"]

        cat_imputer = SimpleImputer(strategy="most_frequent")
        train_df[null_cat_columns] = cat_imputer.fit_transform(train_df[null_cat_columns])
        test_df[null_cat_columns] = cat_imputer.transform(test_df[null_cat_columns])
        
        null_numeric_columns = ["Supplier_Reliability","Equipment_Height","Equipment_Width","Equipment_Weight"]

        num_imputer = SimpleImputer(strategy="median")
        train_df[null_numeric_columns] = num_imputer.fit_transform(train_df[null_numeric_columns])
        test_df[null_numeric_columns] = num_imputer.transform(test_df[null_numeric_columns])
        
        
        # 2. Feature Engineering
        date_columns = ["Order_Placed_Date","Delivery_Date"]
        for df in [train_df, test_df]:
            for column in date_columns:
                df[column] = pd.to_datetime(df[column],format='%m/%d/%y')
            
            for column in date_columns:
                df[column+"_day"] = df[column].dt.day
                df[column+"_month"] = df[column].dt.month
                df[column+"_weekday"] = df[column].dt.dayofweek
                df[column+"_day_of_year"] = df[column].dt.dayofyear
                df[column+"_is_weekend"] = (df[column].dt.dayofweek >= 5).astype(int)
                
            df["Delivery_Days"] = ((df["Delivery_Date"] - df["Order_Placed_Date"]).dt.total_seconds()/86400.0)
            df["Delivery_Days"] = df["Delivery_Days"].clip(lower=0)
            
            df['Value_per_weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1e-6)
            df['Fee_per_weight'] = df['Base_Transport_Fee'] / (df['Equipment_Weight'] + 1e-6)
            
            df['value_to_fee_ratio'] = df['Equipment_Value'] / (df['Base_Transport_Fee'] + 1e-6)
            df['delivery_days_per_reliability'] = df['Delivery_Days'] / (df['Supplier_Reliability'] + 1e-6)
            df['value_per_day'] = df['Equipment_Value'] / (df['Delivery_Days'] + 1)
            df['fragile_and_urgent'] = (df['Fragile_Equipment'] == 'yes') & (df['Urgent_Shipping'] == 'yes').astype(int)
            df['crossborder_and_urgent'] = (df['CrossBorder_Shipping'] == 'yes') & (df['Urgent_Shipping'] == 'yes').astype(int)        

            df['Hospital_State'] = df['Hospital_Location'].apply(lambda x: x.split(', ')[-1].split(' ')[0] if ', ' in x else 'unknown')
        
        train_df = train_df.drop(columns=["Hospital_Location"],axis=1)
        test_df = test_df.drop(columns=["Hospital_Location"],axis=1)
        
        #  aggregation features on some categorical columns 
        agg_features = {
            'Supplier_Name': ['Supplier_Reliability', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Days'],
            'Equipment_Type': ['Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee'],
            'Hospital_State': ['Supplier_Reliability','Delivery_Days', 'Base_Transport_Fee', 'Equipment_Value']
        }
        
        agg_columns = []
        
        for group_col, features in agg_features.items():
            for feat in features:
                for agg_type in ['mean', 'std', 'max']:
                    new_col_name = f'{feat}_{agg_type}_by_{group_col}'
                    
                    train_df[new_col_name] = train_df.groupby(group_col)[feat].transform(agg_type)
                    
                    stats_map = train_df.groupby(group_col)[feat].agg(agg_type).to_dict()
                    test_df[new_col_name] = test_df[group_col].map(stats_map)
                    
                    fill_value = train_df[feat].agg(agg_type)
                    test_df[new_col_name] = test_df[new_col_name].fillna(fill_value)
                    
                    train_df[new_col_name] = train_df[new_col_name].fillna(fill_value)
                    
                    agg_columns.append(new_col_name)
        

        # 3. Reduce Outliers

        outlier_cols = [
            'Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight',
            'Equipment_Value', 'Base_Transport_Fee','Delivery_Days', 'Value_per_weight','Fee_per_weight',
            'value_to_fee_ratio','delivery_days_per_reliability','value_per_day'
        ]
        outlier_cols.extend([col for col in train_df.columns if '_by_' in col])
        outlier_cols = sorted(list(set(outlier_cols)))
        
        for c in outlier_cols:
            if c in train_df.columns:
                lower = train_df[c].quantile(0.005)
                upper = train_df[c].quantile(0.995)
                train_df[c] = train_df[c].clip(lower, upper)
                test_df[c] = test_df[c].clip(lower, upper)
            
        y = np.log1p(train_df[target_column].clip(lower=0))
        train_df = train_df.drop(columns=target_column)
        
        # 4. Encode the columns
        
        high_cardinality_columns = [
            'Hospital_Id', 'Supplier_Name','Order_Placed_Date', 'Delivery_Date',
            'Hospital_State'
        ]

        low_cardinality_columns = [
            'Equipment_Type', 'CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 
            'Transport_Method', 'Fragile_Equipment', 'Hospital_Info', 'Rural_Hospital' ]
        
        target_encoded_columns = []
        for column in high_cardinality_columns:
            oof_enc, test_enc = self.target_encode_cv(train_df[column], y, test_df[column], n_splits=20, smoothing=40,)
            train_df[column + "_te"] = oof_enc.values
            test_df[column + "_te"] = test_enc.values
            train_df = train_df.drop(columns=[column])
            test_df = test_df.drop(columns=[column])
            target_encoded_columns.append(column+"_te")
            
        train_df = pd.get_dummies(train_df, columns=low_cardinality_columns, drop_first=True)
        test_df = pd.get_dummies(test_df, columns=low_cardinality_columns, drop_first=True)
        train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
        

        # 5. Scale the data in numeric columns
        
        numeric_columns = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight',
            'Equipment_Value', 'Base_Transport_Fee','Delivery_Days', 'Value_per_weight','Fee_per_weight',
            'value_to_fee_ratio','delivery_days_per_reliability','value_per_day']

        numeric_transformer = Pipeline(steps=[
            ('pt', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('scaler', StandardScaler())
        ])
        
        scaler = ColumnTransformer(
            transformers=[
                ('scale_pipeline', numeric_transformer, numeric_columns+agg_columns+target_encoded_columns)
            ],
            remainder='passthrough'
        )
        
        X = scaler.fit_transform(train_df)
        X_test = scaler.transform(test_df)
        
        self.columns = train_df.columns.tolist()
        self.target_column = target_column
        self.numeric_columns = numeric_columns
        self.X = X
        self.y = y
        self.X_test = X_test
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.scaler = scaler