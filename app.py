
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="wide")

model = joblib.load("used_car_price_model.pkl")

st.title("🚗 Used Car Price Estimator")
st.write("Estimate a used car's resale price in India.")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Car name", "Maruti Swift Dzire VDI")
    year = st.number_input("Manufacturing year", 1990, 2026, 2018)
    km_driven = st.number_input("Kilometres driven", 0, 1000000, 50000)
    fuel = st.selectbox("Fuel type", ["Diesel", "Petrol", "LPG", "CNG"])
    seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

with col2:
    owner = st.selectbox("Owner history", [
        "First Owner", "Second Owner", "Third Owner",
        "Fourth & Above Owner", "Test Drive Car"
    ])
    mileage = st.number_input("Mileage (kmpl)", 5.0, 40.0, 20.0)
    engine = st.number_input("Engine (CC)", 500.0, 5000.0, 1200.0)
    max_power = st.number_input("Maximum power (bhp)", 20.0, 500.0, 80.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])
    asking_price = st.number_input("Seller's asking price (₹)", 0, 100000000, 500000, step=10000)

if st.button("Estimate Car Price", type="primary"):
    car = pd.DataFrame([{
        "name": name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": float(seats)
    }])

    predicted_price = max(0, float(model.predict(car)[0]))

    st.subheader("Estimated Resale Price")
    st.success(f"₹{predicted_price:,.0f}")
    st.caption("This is a model estimate, not a guaranteed market price.")

    if asking_price > 0:
        difference = asking_price - predicted_price

        if difference > predicted_price * 0.05:
            st.warning(f"The asking price is ₹{difference:,.0f} above the estimate.")
        elif difference < -predicted_price * 0.05:
            st.info(f"The asking price is ₹{abs(difference):,.0f} below the estimate.")
        else:
            st.success("The asking price is within 5% of the estimate.")

    st.subheader("What influences the model?")

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": names,
        "Importance": regressor.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    st.bar_chart(importance.set_index("Feature"))