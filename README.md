# IoT – PLC Failure Prediction  

## 📌 Project Overview

This project focuses on predictive maintenance in industrial IoT environments.  
Using PLC sensor data, a machine learning model was developed to predict potential system failures **10 minutes before occurrence**.

The goal is to minimize missed failures (false negatives), as undetected risks are more costly in industrial systems.

---

## 🏭 Problem Definition

Industrial PLC systems generate continuous sensor data such as:

- Cycle Time
- Vibration
- Temperature
- Pressure

We define a binary risk label:

RISK = 1 if failure occurs within the next 10 minutes  
RISK = 0 otherwise

This transforms the problem into a time-series binary classification task.

---

## ⚙️ Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
