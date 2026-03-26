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
    return df, rfm

df, rfm = load_data()

st.title("🛒 E-commerce Analytics Dashboard")

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Menu",
    ["📊 Dashboard", "👥 Segmentation", "🎯 Recommendation", "🛍️ Market Basket", "🔮 Prediction", "⚙️ Admin"]
)

# =========================
# DASHBOARD (PLOTLY)
# =========================
if menu == "📊 Dashboard":
    st.subheader("Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", df["order_id"].nunique())
    col2.metric("Customers", df["customer_unique_id"].nunique())
    col3.metric("Revenue", f"${df['payment_value'].sum():,.0f}")

    st.divider()

    # Revenue by category
    top_cat = df.groupby("product_category_name_english")["payment_value"].sum().sort_values(ascending=False).head(10)
    fig = px.bar(top_cat, title="Top Categories by Revenue")
    st.plotly_chart(fig, use_container_width=True)

    # Orders over time
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    time_df = df.groupby(df["order_purchase_timestamp"].dt.date)["order_id"].count()

    fig2 = px.line(time_df, title="Orders Over Time")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# SEGMENTATION (PLOTLY)
# =========================
elif menu == "👥 Segmentation":
    st.subheader("Customer Segmentation (RFM)")

    k = st.slider("Number of clusters", 2, 8, 4)

    model = KMeans(n_clusters=k)
    rfm["cluster"] = model.fit_predict(rfm[["Recency","Frequency","Monetary"]])

    fig = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color=rfm["cluster"].astype(str),
        title="Customer Segments"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# RECOMMENDATION 
# =========================
elif menu == "🎯 Recommendation":
    st.subheader("🎯 Smart Recommendation")

    user_id = st.text_input("Enter Customer ID")

    if user_id:
        user_data = df[df["customer_unique_id"] == user_id]

        if user_data.empty:
            st.warning("User mới → recommend phổ biến")

            rec = (
                df.groupby("product_id")
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
                .agg({
                    "review_score":"mean",
                    "price":"mean"
                })
                .reset_index()
            )

            product_profile = (
                df.groupby(["product_id","product_category_name_english"])
                .agg({
                    "review_score":"mean",
                    "price":"mean"
                })
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
            # UI GRID CARD
            # =========================
            cols = st.columns(2)

            for i, row in rec.iterrows():
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="
                        padding:15px;
                        border-radius:15px;
                        background:white;
                        margin-bottom:15px;
                        box-shadow:0 4px 12px rgba(0,0,0,0.1);
                        border-left:5px solid #4CAF50;
                    ">
                        <h4>🛍️ {row['product_id']}</h4>
                        <p>📦 {row['product_category_name_english']}</p>
                        <p>💰 Price: ${row['price']:.2f}</p>
                        <p>⭐ Rating: {row['review_score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

# =========================
# FP-GROWTH
# =========================
elif menu == "🛍️ Market Basket":
    st.subheader("Association Rules")

    try:
        rules = pd.read_csv("rules.csv")
        st.dataframe(rules.sort_values(by="lift", ascending=False).head(20))
    except:
        st.warning("Run FP-Growth first to generate rules.csv")

# =========================
# PREDICTION
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
# ADMIN
# =========================
elif menu == "⚙️ Admin":
    st.subheader("Admin Panel")

    file = st.file_uploader("Upload new dataset")

    if file is not None:
        new_df = pd.read_csv(file)
        st.write(new_df.head())

    if st.button("Retrain Model"):
        from sklearn.ensemble import RandomForestRegressor

        data_model = df[["price", "freight_value", "payment_value", "review_score"]].dropna()

        X = data_model[["price", "freight_value", "payment_value"]]
        y = data_model["review_score"]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        joblib.dump(model, "classifier.pkl")

        st.success("Model retrained!")
