import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model(path="model.pkl"):
    """Load the pre-trained model. Accept either a raw model or a dict containing 'model'."""
    try:
        loaded = joblib.load(path)
        # support both saving styles: either direct model or dict {"model": model, ...}
        if isinstance(loaded, dict) and "model" in loaded:
            return loaded["model"]
        return loaded
    except FileNotFoundError:
        st.error(f"Error: '{path}' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def get_data_info(csv_path="Airlines_Flights_Data.csv"):
    """
    Load the CSV and extract:
      - feature_order: the order of columns used for prediction (X columns)
      - categorical_cols, numerical_cols
      - unique_values for categorical dropdowns
      - mappings: value -> integer encoding (LabelEncoder-like)
      - num_stats: min/max/mean for numeric inputs (to set defaults)
    """
    try:
        df = pd.read_csv(csv_path)

        # Columns dropped during training (keep consistent with your training script)
        cols_to_drop = ['index', 'flight', 'arrival_time', 'duration', 'days_left']
        existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        # Ensure 'price' (target) exists
        if 'price' not in df.columns:
            raise KeyError("Missing 'price' column in CSV.")

        # Drop rows with missing price and coerce numeric just in case
        df = df.dropna(subset=['price'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])

        # Prepare X (features) - keep the same order as training
        X = df.drop('price', axis=1)
        feature_order = X.columns.tolist()

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        unique_values = {}
        mappings = {}
        # Create mapping consistent with a LabelEncoder fitted on the dataset values.
        # We sort unique values so the mapping is deterministic.
        for col in categorical_cols:
            vals = df[col].astype(str).unique().tolist()
            vals_sorted = sorted(vals)
            le = LabelEncoder()
            le.fit(vals_sorted)  # classes_ will be in this deterministic order
            unique_values[col] = vals_sorted
            mappings[col] = {label: int(idx) for idx, label in enumerate(le.classes_)}

        # Numeric stats for better UI defaults
        num_stats = {}
        for col in numerical_cols:
            col_series = X[col].dropna().astype(float)
            if len(col_series) == 0:
                num_stats[col] = {"min": 0.0, "max": 100.0, "mean": 0.0}
            else:
                num_stats[col] = {
                    "min": float(col_series.min()),
                    "max": float(col_series.max()),
                    "mean": float(col_series.mean())
                }

        return feature_order, categorical_cols, numerical_cols, unique_values, mappings, num_stats

    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. The dataset is required for generating inputs.")
        return None, None, None, None, None, None
    except KeyError as e:
        st.error(f"Critical CSV error: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading CSV: {e}")
        return None, None, None, None, None, None

# --- Load Model and Data Info ---
model = load_model("model.pkl")
data_info = get_data_info("Airlines_Flights_Data.csv")

if data_info[0] is not None:
    feature_order, categorical_cols, numerical_cols, unique_values, mappings, num_stats = data_info
else:
    feature_order, categorical_cols, numerical_cols, unique_values, mappings, num_stats = [], [], [], {}, {}, {}

# --- Main UI ---
st.title("✈️ Decision Tree Flight Price Predictor")
st.markdown("Provide the flight details in the sidebar to get an estimated price.")

if model is None:
    st.warning("Model not loaded. Fix model.pkl path or recreate the model file and place it here.")
elif not feature_order:
    st.warning("Dataset info not available. Fix CSV path or ensure it contains the same columns used during training.")
else:
    st.sidebar.header("Flight Details")

    with st.sidebar.form(key='prediction_form'):
        inputs = {}
        # Build inputs in the same order as training
        for col in feature_order:
            label = col.replace('_', ' ').title()
            if col in categorical_cols:
                # Dropdown of unique (string) values from training CSV
                options = unique_values.get(col, [])
                # If no options, create an empty dropdown to avoid KeyError later
                if options:
                    inputs[col] = st.selectbox(label, options=options)
                else:
                    inputs[col] = st.text_input(label, value="")
            elif col in numerical_cols:
                stats = num_stats.get(col, {"min": 0.0, "max": 100.0, "mean": 0.0})
                # Provide a number input with sensible min/max and mean as default
                # If these values are floats, we'll use step=0.01, else default
                default_val = stats["mean"]
                min_val = stats["min"]
                max_val = stats["max"]
                # Choose step based on the data spread
                step = 1.0 if float(max_val - min_val) > 10 else 0.01
                try:
                    inputs[col] = st.number_input(label, value=float(default_val), min_value=float(min_val), max_value=float(max_val), step=step)
                except Exception:
                    # fallback in case of problematic bounds
                    inputs[col] = st.number_input(label, value=0.0)

        submit_button = st.form_submit_button(label='Predict Price', use_container_width=True)

    if submit_button:
        try:
            # Build feature vector using the exact order used in training
            feature_vector = []
            for col in feature_order:
                val = inputs[col]
                if col in categorical_cols:
                    # encode using mapping built from training CSV
                    mapping = mappings.get(col, {})
                    # handle case user typed something not in mapping
                    if val not in mapping:
                        st.warning(f"Value '{val}' for '{col}' was not seen during training. Attempting to map, otherwise using 0.")
                        enc = mapping.get(str(val), 0)
                    else:
                        enc = mapping[val]
                    feature_vector.append(enc)
                else:
                    # numerical - ensure it's a numeric type
                    try:
                        feature_vector.append(float(val))
                    except Exception:
                        feature_vector.append(0.0)

            features_array = np.array(feature_vector).reshape(1, -1)

            # Prediction
            predicted = model.predict(features_array)
            predicted_price = float(predicted[0])

            # Show result
            st.subheader("Predicted Flight Price:")
            st.metric(label="Estimated Cost", value=f"₹ {predicted_price:,.2f}")
            st.info("Disclaimer: This is an estimated price based on the trained ML model and the provided dataset. Actual prices may vary.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
