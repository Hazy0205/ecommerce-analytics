import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.cluster import KMeans

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="E-commerce Analytics", layout="wide")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data_small.csv")
    rfm = pd.read_csv("rfm_data.csv")

    # 🔥 FIX STRING
    df["customer_unique_id"] = df["customer_unique_id"].astype(str).str.strip()

    return df, rfm

df, rfm = load_data()

st.title("🛒 E-commerce Analytics Dashboard")

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Menu",
    ["📊 Dashboard", "👥 Segmentation", "🎯 Recommendation", "🔮 Prediction", "⚙️ Admin"]
)

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":
    st.subheader("Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", df["order_id"].nunique())
    col2.metric("Customers", df["customer_unique_id"].nunique())
    col3.metric("Revenue", f"${df['payment_value'].sum():,.0f}")

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":
    st.subheader("Customer Segmentation")

    k = st.slider("Clusters", 2, 6, 3)
    model = KMeans(n_clusters=k)
    rfm["cluster"] = model.fit_predict(rfm[["Recency","Frequency","Monetary"]])

    st.scatter_chart(rfm, x="Recency", y="Monetary")

# =========================
# RECOMMENDATION (FIX)
# =========================
elif menu == "🎯 Recommendation":
    st.subheader("🎯 Smart Recommendation")

    user_id = st.text_input("Enter Customer ID")

    if user_id:
        user_id = user_id.strip()

        user_data = df[df["customer_unique_id"] == user_id]

        if user_data.empty:
            st.warning("User mới → recommend phổ biến")

            rec = (
                df.groupby(["product_id","product_category_name_english"])
                .agg({"review_score":"count","price":"mean"})
                .reset_index()
                .sort_values(by="review_score", ascending=False)
                .head(10)
            )

            st.dataframe(rec)

        else:
            st.success("Personalized recommendations")

            user_profile = (
                user_data.groupby("product_category_name_english")
                .agg({"review_score":"mean"})
                .reset_index()
            )

            product_profile = (
                df.groupby(["product_id","product_category_name_english"])
                .agg({"review_score":"mean","price":"mean"})
                .reset_index()
            )

            merged = product_profile.merge(
                user_profile,
                on="product_category_name_english",
                suffixes=("_prod","_user")
            )

            merged["score"] = (
                merged["review_score_prod"] * 0.7 +
                merged["review_score_user"] * 0.3
            )

            bought = user_data["product_id"].unique()
            merged = merged[~merged["product_id"].isin(bought)]

            rec = merged.sort_values(by="score", ascending=False).head(10)

            st.dataframe(rec[["product_id","score","price_prod"]])

# =========================
# PREDICTION (FIX)
# =========================
elif menu == "🔮 Prediction":
    st.subheader("Predict Review Score")

    price = st.number_input("Price", 0.0)
    freight = st.number_input("Freight", 0.0)
    payment = st.number_input("Payment", 0.0)

    if st.button("Predict"):
        try:
            model = joblib.load("classifier.pkl")
            pred = model.predict([[price, freight, payment]])
            st.success(f"Predicted Score: {round(pred[0],2)}")
        except:
            st.error("Train model first")

# =========================
# ADMIN (FIX)
# =========================
elif menu == "⚙️ Admin":
    st.subheader("Admin Panel")

    file = st.file_uploader("Upload dataset (.csv or .xlsx)")

    if file is not None:
        try:
            if file.name.endswith(".csv"):
                new_df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                new_df = pd.read_excel(file)
            else:
                st.error("Unsupported file")
                st.stop()

            st.success("Upload thành công")
            st.write(new_df.head())

        except Exception as e:
            st.error(f"Lỗi: {e}")

    if st.button("Retrain Model"):
        from sklearn.ensemble import RandomForestRegressor

        data_model = df[["price","freight_value","payment_value","review_score"]].dropna()

        X = data_model[["price","freight_value","payment_value"]]
        y = data_model["review_score"]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        joblib.dump(model, "classifier.pkl")

        st.success("Model retrained!")
