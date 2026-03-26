import joblib
import os
import numpy as np

def extract():
    try:
        meta = joblib.load("modeltraining/ensemble_meta.pkl")
        scaler = joblib.load("modeltraining/meta_scaler.pkl")
        
        print(f"a (Intercept): {meta.intercept_[0]}")
        print(f"w1 (XGB Coeff): {meta.coef_[0][0]}")
        print(f"w2 (LGBM Coeff): {meta.coef_[0][1]}")
        print(f"m1 (XGB Mean): {scaler.mean_[0]}")
        print(f"m2 (LGBM Mean): {scaler.mean_[1]}")
        print(f"s1 (XGB Scale): {scaler.scale_[0]}")
        print(f"s2 (LGBM Scale): {scaler.scale_[1]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract()
