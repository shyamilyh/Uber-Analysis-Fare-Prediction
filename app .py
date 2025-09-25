"""
Streamlit app to deploy the Uber fare prediction model.

How to use:
1. From your notebook save the trained model and preprocessor (if you have one):

   import joblib
   joblib.dump(model, 'model.pkl')  # trained scikit-learn estimator
   joblib.dump(preprocessor, 'preprocessor.pkl')  # optional: preprocessing pipeline

2. Put `app.py`, `model.pkl` (and optionally `preprocessor.pkl`) in the same folder.
3. Install requirements: pip install streamlit pandas numpy scikit-learn joblib
4. Run: streamlit run app.py

This app tries to be flexible: if you saved a preprocessing pipeline (encoder/scaler/feature builder)
it will use it. If not, it will compute a few common features (haversine distance + datetime parts)
and feed them to the model. If your model expects differently-named features, adjust the code where
`X` is assembled to match the feature order/names used when training.

"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, date, time
from math import radians, cos, sin, asin, sqrt

# ---------- Utilities ----------

# Use the same radius as in the notebook for consistency
EARTH_RADIUS_KM = 6367

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two points."""
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = EARTH_RADIUS_KM * c
    return km


def load_artifact(filename):
    if os.path.exists(filename):
        try:
            return joblib.load(filename)
        except Exception:
            try:
                import pickle
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Failed to load {filename}: {e}")
                return None
    return None


# ---------- Load model & preprocessor ----------

MODEL_NAMES = ['model.pkl', 'uber_fare_model.pkl', 'model.joblib']
PREPRO_NAMES = ['preprocessor.pkl', 'pipeline.pkl', 'preprocessor.joblib']

model = None
preprocessor = None

for n in MODEL_NAMES:
    if os.path.exists(n):
        model = load_artifact(n)
        if model is not None:
            # Guard print when not in streamlit context
            try:
                st.write(f"Loaded model from {n}")
            except:
                print(f"Loaded model from {n}")
            break

for n in PREPRO_NAMES:
    if os.path.exists(n):
        preprocessor = load_artifact(n)
        if preprocessor is not None:
             # Guard print when not in streamlit context
            try:
                st.write(f"Loaded preprocessor/pipeline from {n}")
            except:
                print(f"Loaded preprocessor/pipeline from {n}")
            break

# If running on Streamlit server, the st.write used above will appear; but on import it may error.
# To avoid issues when Streamlit isn't yet running, guard prints when not in streamlit.

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Uber Fare Predictor", layout="centered")
st.title("Uber / Taxi Fare Prediction")
st.markdown(
    "Enter trip details (pickup/dropoff coords, datetime, passenger count). The app will compute distance & time features and show the predicted fare."
)

with st.sidebar:
    st.header("Input options")
    use_sample = st.checkbox("Use sample values (NYC)", value=True)

if use_sample:
    # reasonable defaults for a NYC-style trip
    default_date = date(2019, 1, 1)
    default_time = time(8, 30)
    default_plon = -73.985428
    default_plat = 40.748817
    default_dlon = -73.9753
    default_dlat = 40.7527
    default_pass = 1
else:
    default_date = date.today()
    default_time = time(datetime.now().hour, datetime.now().minute)
    default_plon = 0.0
    default_plat = 0.0
    default_dlon = 0.0
    default_dlat = 0.0
    default_pass = 1

col1, col2 = st.columns(2)
with col1:
    pickup_date = st.date_input("Pickup date", value=default_date)
    pickup_time = st.time_input("Pickup time", value=default_time)
    passenger_count = st.number_input("Passenger count", min_value=1, max_value=10, value=default_pass)

with col2:
    pickup_latitude = st.number_input("Pickup latitude", value=default_plat, format="%.6f")
    pickup_longitude = st.number_input("Pickup longitude", value=default_plon, format="%.6f")
    dropoff_latitude = st.number_input("Dropoff latitude", value=default_dlat, format="%.6f")
    dropoff_longitude = st.number_input("Dropoff longitude", value=default_dlon, format="%.6f")

st.markdown("---")

# Combine date and time
pickup_datetime_combined = datetime.combine(pickup_date, pickup_time)

# Calculate distance
distance_km = haversine_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

# Approximate coordinates for Times Square, NYC - Use same as in notebook
times_square_lat = 40.7579
times_square_lon = -73.9855
pickup_distance_to_times_square = haversine_distance(pickup_latitude, pickup_longitude, times_square_lat, times_square_lon)
dropoff_distance_to_times_square = haversine_distance(dropoff_latitude, dropoff_longitude, times_square_lat, times_square_lon)


st.metric(label="Haversine distance (km)", value=f"{distance_km:.3f} km")

# Build features dataframe with names and order matching training data
# The feature list from the notebook is:
# features = ['passenger_count', 'year', 'month', 'day_of_week', 'hour_of_day',
#             'distance', 'day_of_year', 'week_of_year', 'is_weekend',
#             'pickup_distance_to_times_square', 'dropoff_distance_to_times_square']

dt = pd.to_datetime(pickup_datetime_combined)

feature_data = {
    'passenger_count': [int(passenger_count)],
    'year': [dt.year],
    'month': [dt.month],
    'day_of_week': [dt.weekday()], # Monday=0, Sunday=6
    'hour_of_day': [dt.hour],
    'distance': [distance_km], # Note: Using 'distance' to match training feature name
    'day_of_year': [dt.dayofyear],
    'week_of_year': [dt.isocalendar().week], # This returns UInt32, will convert to int later
    'is_weekend': [1 if dt.weekday() >= 5 else 0], # Check if weekday is 5 (Saturday) or 6 (Sunday)
    'pickup_distance_to_times_square': [pickup_distance_to_times_square],
    'dropoff_distance_to_times_square': [dropoff_distance_to_times_square]
}

# Define the correct order of features based on the notebook
trained_features_order = ['passenger_count', 'year', 'month', 'day_of_week', 'hour_of_day',
                          'distance', 'day_of_year', 'week_of_year', 'is_weekend',
                          'pickup_distance_to_times_square', 'dropoff_distance_to_times_square']

# Create DataFrame and ensure column order matches trained features
X = pd.DataFrame(feature_data)[trained_features_order]

# Ensure week_of_year is int to match training data type
X['week_of_year'] = X['week_of_year'].astype(int)


st.write("### Features sent to model")
st.dataframe(X.T)

# Try prediction
predict_button = st.button("Predict fare")

if predict_button:
    if model is None:
        st.error(
            "No model found in the working directory. Please save your trained model as 'model.pkl' or 'uber_fare_model.pkl' next to app.py."
        )
    else:
        try:
            # If a preprocessor/pipeline exists, try applying it first
            if preprocessor is not None:
                try:
                    # Preprocessor transform should handle feature names and order
                    X_trans = preprocessor.transform(X)
                except Exception as e:
                    st.warning(f"Preprocessor exists but failed to transform input: {e}. Trying to pass raw dataframe to model.")
                    X_trans = X
                preds = model.predict(X_trans)
            else:
                # Attempt to feed model the DataFrame directly.
                # Feature names and order should now match due to previous steps.
                preds = model.predict(X)


            predicted_fare = float(preds[0])
            if predicted_fare < 0:
                predicted_fare = abs(predicted_fare)
            st.success(f"Predicted fare: ${predicted_fare:.2f}")
            st.info("Note: If the prediction looks odd, check that the model was trained with the same features & units (distance in km vs miles, etc.)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

st.write("## Help & Troubleshooting")
st.write(
    "If your model expects a different set of features, edit the code where the `X` DataFrame is created to match the training features exactly (column names and order).\n\n"
)

st.write("### Example: how you may have saved model & pipeline in your notebook")
st.code(
    """
# after training
import joblib
joblib.dump(best_estimator, 'model.pkl')
joblib.dump(preprocessor_pipeline, 'preprocessor.pkl')
""",
    language='python',
)

st.write("### Notes")
st.write("- If your training used distance in miles, convert units accordingly.\n- If categorical encoders were used (one-hot, ordinal), saving the preprocessor is highly recommended so the app can replicate training-time transformations.")

# End of app
