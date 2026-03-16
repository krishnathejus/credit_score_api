from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load artifacts
woe_table = pd.read_csv("woe_bins.csv")
model = joblib.load("credit_logreg_model.pkl")
Factor = joblib.load("score_factor.pkl")
Offset = joblib.load("score_offset.pkl")
features = joblib.load("model_features.pkl")

scorecard = pd.read_csv("credit_scorecard.csv")

base_score = scorecard[scorecard["VAR_NAME"]=="Base_Score"]["Points"].values[0]

scorecard = scorecard[scorecard["VAR_NAME"]!="Base_Score"]

# -----------------------------------

def get_points(var, value):

    bins = scorecard[scorecard["VAR_NAME"]==var]

    for _, row in bins.iterrows():

        if row["MIN_VALUE"] <= value <= row["MAX_VALUE"]:
            return row["Points"]

    return 0

# -----------------------------
# WOE Transformation Function
# -----------------------------
def apply_woe(df):

    result = pd.DataFrame()

#   for var in woe_table["VAR_NAME"].unique():
    for var in [f.replace("new_","") for f in features]:

        temp = woe_table[woe_table["VAR_NAME"] == var]
        var_type = temp["VAR_TYPE"].iloc[0]

        # ------------------------
        # NUMERIC VARIABLES
        # ------------------------
        if var_type == "Numeric":

            # ensure numeric type
            temp.loc[:, "MIN_VALUE"] = pd.to_numeric(temp["MIN_VALUE"])
            temp.loc[:, "MAX_VALUE"] = pd.to_numeric(temp["MAX_VALUE"])

            def map_bin(x):
                for _, row in temp.iterrows():
                    if row["MIN_VALUE"] <= x <= row["MAX_VALUE"]:
                        return row["WOE"]
                return 0

            result["new_" + var] = df[var].astype(float).apply(map_bin)

        # ------------------------
        # CATEGORICAL VARIABLES
        # ------------------------
        else:

            mapping = dict(zip(temp["MIN_VALUE"], temp["WOE"]))

            result["new_" + var] = df[var].map(mapping).fillna(0)

    return result


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Credit score prediction API"}


@app.post("/predict")
def predict(data: dict):

    # Convert request → dataframe
    df = pd.DataFrame([data])

    # Apply WOE transformation
    df_woe = apply_woe(df)

    # Ensure model feature order
    df_woe = df_woe.reindex(columns=features, fill_value=0)

    # Predict probability of default
    pd_pred = model.predict_proba(df_woe)[:,1][0]
    
    score = base_score
    for var in [f.replace("new_","") for f in features]:

        
        if var not in df.columns:
            continue
            
        bin_table = scorecard[scorecard["VAR_NAME"] == var]

        value = df[var].iloc[0]

        if bin_table["VAR_TYPE"].iloc[0] == "Numeric":

            row = bin_table[
                (pd.to_numeric(bin_table["MIN_VALUE"]) <= value) &
                (value <= pd.to_numeric(bin_table["MAX_VALUE"]))
            ]

        else:

            row = bin_table[bin_table["MIN_VALUE"] == value]

        if len(row) > 0:
            score += row["Points"].values[0]


    risk_band = "Low Risk"
    if score < 550:
        risk_band = "High Risk"
    elif score < 650:
        risk_band = "Medium Risk"

    return {
        "credit_score": round(score, 0),
        "pd": float(pd_pred),
        "risk_band": risk_band
    }
