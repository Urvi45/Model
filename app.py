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
def load_model():
    """Loads the pre-trained Decision Tree model."""
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
        return None

@st.cache_data
def get_data_info():
    """
    Loads the raw dataset to extract feature information, identify categorical
    and numerical columns, and create mappings for the UI and prediction.
    This dynamically mimics the preprocessing from the training script.
    """
    try:
        df = pd.read_csv("Airlines_Flights_Data.csv")

        # 1. Replicate preprocessing from the training script
        # Drop the same columns dropped during training to avoid errors
        cols_to_drop = ['index', 'flight']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df_processed = df.drop(columns=existing_cols_to_drop)

        # 2. Identify feature columns (everything except the target 'price')
        # This ensures the app uses the exact features the model was trained on
        X = df_processed.drop('price', axis=1)
        feature_order = X.columns.tolist() # The exact order needed for prediction

        # 3. Separate columns by data type for generating the correct UI inputs
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

        # 4. Create mappings and unique value lists for dropdowns (for categorical features)
        mappings = {}
        unique_values = {}

        for col in categorical_cols:
            le = LabelEncoder()
            # Ensure consistent encoding by fitting on sorted, string-converted unique values
            unique_vals = sorted(df[col].astype(str).unique())
            le.fit(unique_vals)
            
            unique_values[col] = unique_vals
            mappings[col] = {label: index for index, label in enumerate(le.classes_)}
        
        return feature_order, categorical_cols, numerical_cols, unique_values, mappings

    except FileNotFoundError:
        st.error("Error: 'Airlines_Flights_Data.csv' not found. The dataset is required for the input fields.")
        return None, None, None, None, None
    except KeyError as e:
        st.error(f"A critical column is missing from the dataset: {e}. Please ensure the 'price' column exists and other columns match the training data.")
        return None, None, None, None, None

# --- Load Model and Data Info ---
model = load_model()
data_info = get_data_info()

# Unpack data_info if it's not None
if data_info:
    feature_order, categorical_cols, numerical_cols, unique_values, mappings = data_info
else:
    # Set to empty lists to prevent further errors if loading fails
    feature_order, categorical_cols, numerical_cols, unique_values, mappings = [], [], [], {}, {}

# --- Main Application UI ---
st.title("✈️ Decision Tree Flight Price Predictor")
st.markdown("Provide the flight details in the sidebar to get an estimated price.")

if model and data_info:
    st.sidebar.header("Flight Details")
    
    with st.sidebar.form(key='prediction_form'):
        
        inputs = {} # Dictionary to store all user inputs

        # Create input fields dynamically based on column type
        for col in feature_order:
            label = col.replace('_', ' ').title()
            if col in categorical_cols:
                inputs[col] = st.selectbox(label, options=unique_values[col])
            elif col in numerical_cols:
                # Use a reasonable default for number inputs
                inputs[col] = st.number_input(label, value=0)
        
        submit_button = st.form_submit_button(label='Predict Price', use_container_width=True)

    # --- Prediction Logic ---
    if submit_button:
        try:
            # Construct the feature vector in the correct order for the model
            feature_vector = []
            for col in feature_order:
                value = inputs[col]
                if col in categorical_cols:
                    # Use the saved mapping to encode the categorical feature
                    feature_vector.append(mappings[col][value])
                else: # Numerical feature
                    feature_vector.append(value)
            
            # Reshape for a single prediction
            features_array = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_array)
            predicted_price = prediction[0]

            # --- Display Result ---
            st.subheader("Predicted Flight Price:")
            st.metric(label="Estimated Cost", value=f"₹ {predicted_price:,.2f}")
            st.info("Disclaimer: This is an estimated price based on the machine learning model and may not reflect the actual market price.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Application could not start. Please check for error messages above regarding missing files.")
