import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import joblib



# 1. Build preprocessing pipeline

def build_pipelines(num_attributes, cat_attributes):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)
    ])

    return full_pipeline



# 2. Load dataset
if not os.path.exists(("housing_model.pkl") and ("pipeline.pkl")):

    data = pd.read_csv("housing.csv")



    # 3. Create income category for stratified sampling

    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )




    # 4. Stratified train-test split

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data["income_cat"]):
        train_set = data.loc[train_index].drop("income_cat", axis=1)
        test_set = data.loc[test_index].drop("income_cat", axis=1)





    # 5. Separate features and labels

    train_labels = train_set["median_house_value"].copy()
    train_features = train_set.drop("median_house_value", axis=1)





    # 6. Identify numerical and categorical attributes

    num_attributes = train_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]





    # 7. Apply preprocessing pipeline

    full_pipeline = build_pipelines(num_attributes, cat_attributes)
    train_prepared = full_pipeline.fit_transform(train_features)





    # 8. Train Linear Regression

    lin_model = LinearRegression()
    lin_model.fit(train_prepared, train_labels)

    lin_predictions = lin_model.predict(train_prepared)

    lin_rmse = np.sqrt(mean_squared_error(train_labels, lin_predictions))

    print("Linear Regression RMSE:", lin_rmse)

    lin_rmses = -cross_val_score(
        lin_model,
        train_prepared, 
        train_labels, 
        scoring="neg_root_mean_squared_error", 
        cv=10
    )
    # print(f"The root mean squared error for Linear Regression is {lin_rmses}")
    print(pd.Series(lin_rmses).describe())





    # 9. Train Decision Tree

    dec_reg = DecisionTreeRegressor()
    dec_reg.fit(train_prepared, train_labels)
    dec_preds = dec_reg.predict(train_prepared)
    # dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
    dec_rmses = -cross_val_score(
        dec_reg,
        train_prepared,
        train_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    # print(f"The root mean squared error for Decision Tree is {dec_rmses}")
    print(pd.Series(dec_rmses).describe())






    # 10. Train Random Forest

    forest_model = RandomForestRegressor(random_state=42)
    forest_model.fit(train_prepared, train_labels)

    forest_predictions = forest_model.predict(train_prepared)

    forest_rmse = np.sqrt(mean_squared_error(train_labels, forest_predictions))

    print("Random Forest RMSE:", forest_rmse)

    rndm_frst_scores = -cross_val_score(
        forest_model,
        train_prepared,
        train_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )

    print(pd.Series(rndm_frst_scores).describe())





    # 11. Save trained model

    joblib.dump(forest_model, "housing_model.pkl")
    joblib.dump(full_pipeline, "pipeline.pkl")
    print("Model and pipeline saved")

else:
    model = joblib.load("housing_model.pkl")
    pipeline = joblib.load("pipeline.pkl")
 
    # input_data = pd.read_csv("input.csv")
    # transformed_input = pipeline.transform(input_data)
    # predictions = model.predict(transformed_input)
    # input_data["median_house_value"] = predictions
 
    # input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")