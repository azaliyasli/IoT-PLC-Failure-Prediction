import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# --- Preprocessing ---
df = pd.read_csv("plc_signal_failure_data.csv")

df = df.dropna()

features = ["CYCLE_TIME", "VIBRATION", "TEMPERATURE", "PRESSURE"]
df["RISK"] = (df["MINUTES_TO_FAILURE"] <= 10).astype(int)

X = df[features]
y = df["RISK"]

# --- Class Weight ---
neg = (y == 0).sum()
pos = (y == 1).sum()
scale_pos_weight = neg / pos

# --- Cross Validation ---
tscv = TimeSeriesSplit(n_splits=5)

roc_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    )

    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.45).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    roc_scores.append(roc)

    print(f"\nFold {fold + 1}")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc)

print("\n--------------------------------")
print("Mean ROC-AUC:", np.mean(roc_scores))