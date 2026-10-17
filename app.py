import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# -----------------------------
# 2. Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Keep house_price_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

# -----------------------------
# 3. App Header
# -----------------------------
st.title("🏡 California House Price Predictor")

st.write(
    "Enter the housing details below to estimate "
    "the median house value using our trained Ridge Regression model."
)

st.info(
    "This model uses the California Housing dataset. "
    "The prediction is an estimate, not an actual market quote."
)

# -----------------------------
# 4. User Input Form
# -----------------------------
st.subheader("Enter Housing Details")

with st.form("house_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income (in $10,000 units)",
            min_value=0.0,
            max_value=20.0,
            value=3.5,
            step=0.1,
            help="For example, 3.5 represents approximately $35,000."
        )

        house_age = st.number_input(
            "House Age (years)",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=1.0
        )

        ave_rooms = st.number_input(
            "Average Rooms per Household",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms per Household",
            min_value=0.5,
            max_value=50.0,
            value=1.0,
            step=0.1
        )

    with col2:
        population = st.number_input(
            "Block Population",
            min_value=1.0,
            max_value=50000.0,
            value=1000.0,
            step=100.0
        )

        ave_occup = st.number_input(
            "Average Occupants per Household",
            min_value=0.5,
            max_value=100.0,
            value=3.0,
            step=0.1
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.5,
            value=36.0,
            step=0.1
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-119.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "Predict House Value",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# 5. Make Prediction
# -----------------------------
if submitted:

    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    try:
        predicted_value = float(model.predict(input_data)[0])

        # Dataset target is measured in $100,000 units
        predicted_dollars = predicted_value * 100000

        st.divider()
        st.subheader("Prediction Result")

        if predicted_value < 0:
            st.warning(
                "The model returned a negative estimate. "
                "This is not a meaningful house price; "
                "consider checking the inputs and model limitations."
            )
        else:
            st.metric(
                label="Estimated Median House Value",
                value=f"${predicted_dollars:,.0f}"
            )

            st.caption(
                "The dataset target is capped at approximately "
                "$500,000, so predictions near or above this level "
                "should be interpreted cautiously."
            )

            with st.expander("View the input values"):
                st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# -----------------------------
# 6. Footer
# -----------------------------
st.divider()
st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit | "
    "Machine Learning Project"
)
