# Project objective

**Project:** Uber Fare Analysis & Prediction

**Summary:** This project analyzes Uber trip data and builds a regression model to predict fare amounts for individual rides. The repository contains the notebook, feature utilities, training scripts, and a Streamlit demo for inference (if present).

## Dataset

Data files referenced in the notebook:

- `/content/uber.csv`

**Typical columns expected:** `pickup_datetime`, `dropoff_datetime` (or duration), `pickup_lat`, `pickup_long`, `dropoff_lat`, `dropoff_long`, `passenger_count`, `fare_amount` (target).

## Feature Engineering

- Haversine distance function detected and used to compute trip distance (km).

- Datetime decomposition (hour, weekday, month), distance / duration derived features, interaction features, and outlier flags detected or recommended.

## Modeling

- Models implemented: CatBoostRegressor, GradientBoostingRegressor, LGBMRegressor, LinearRegression, RandomForestRegressor, XGBRegressor.

- Train/test split present: Yes; Cross-validation/tuning present: Yes.

## Evaluation Metrics

 Results for Tuned Random Forest Regressor:

 
Model	                     |   MSE	      |    RMSE   |	 R-squared  |
_____________________________________________________________________

Linear Regression	         |    27.8532	  |   5.2776	|   0.7130    |
Random Forest (Untuned)	   |    18.9585	  |   4.3541	|   0.8046    |
Gradient Boosting 	       |    19.2816	  |   4.3911	|   0.8013    |
XGBoost 	                 |    20.7825	  |   4.5588	|   0.7858    |
LightGBM 	                 |    19.9279	  |   4.4641	|   0.7946    |
CatBoost 	                 |    20.8648   |   4.5678	|   0.7850    |
Random Forest (Tuned)	     |    18.7024	  |   4.3246	|   0.8073    |

## Final Results 

- **Best model:** `Tuned Random Forest Regressor`

  MSE: 18.7024
  RMSE: 4.3246
  R-squared: 0.8073

## Deployment / App

- Model deployment in Streamlit include `app.py` in repo and the commands to run it.

## How to run

1. Create virtual environment and install dependencies from `requirements.txt`.

2. Launch the notebook: `jupyter lab notebooks/Uber_Analysis_and_Fare_Prediction.ipynb` and run cells sequentially.

3. To train a model from scripts : `python src/train.py --data data/uber_trips.csv --output models/`.

4. To run the Streamlit demo: `streamlit run app.py`.

## Repository Structure

```
uber-fare-prediction/
├─ notebooks/
│  └─ Uber_Analysis_and_Fare_Prediction.ipynb
├─ src/
│  ├─ features.py
│  ├─ train.py
│  └─ predict.py
├─ app/
│  └─ app.py  # optional Streamlit demo
├─ models/
│  └─ final_model.pkl
├─ data/
│  └─ uber_trips.csv
├─ requirements.txt
└─ README.md
```

## Contact

- Author: `Shyamily`
- GitHub link: 
- Streamlit link
