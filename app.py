import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-commerce Analytics", layout="wide")

# =========================
# LOAD DATA
# =========================
def load_data(file=None):
    if file:
        return pd.read_csv(file)
    return pd.read_csv("cleaned_data_small.csv")

# =========================
# RFM FUNCTION (CLEAN + SAFE)
# =========================
def create_rfm(data):
    rfm = data.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": "max",
        "order_id": "count",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_id","Recency","Frequency","Monetary"]

    # Convert Recency
    rfm["Recency"] = (pd.to_datetime("today") - pd.to_datetime(rfm["Recency"], errors='coerce')).dt.days

    # Convert numeric
    for col in ["Recency","Frequency","Monetary"]:
        rfm[col] = pd.to_numeric(rfm[col], errors="coerce")

    # Clean data
    rfm = rfm.replace([np.inf, -np.inf], np.nan)
    rfm = rfm.dropna(subset=["Recency","Frequency","Monetary"])

    return rfm

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Menu",
    [
        "📊 Dashboard",
        "👥 Segmentation",
        "🎯 Recommendation",
        "🛍️ Market Basket",
        "🔮 Prediction",
        "⚙️ Admin",
    ],
)

# =========================
# DATA
# =========================
df = load_data()

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", df["order_id"].nunique())
    col2.metric("Customers", df["customer_unique_id"].nunique())
    col3.metric("Revenue", f"${df['payment_value'].sum():,.0f}")

    st.divider()

    # Charts
    top_cat = df.groupby("product_category_name_english")["payment_value"].sum().sort_values(ascending=False).head(10)
    st.plotly_chart(px.bar(top_cat, title="Top Categories"), use_container_width=True)

    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
    time_df = df.groupby(df["order_purchase_timestamp"].dt.date)["order_id"].count()
    st.plotly_chart(px.line(time_df, title="Orders Over Time"), use_container_width=True)

    # Clustering
    st.subheader("Customer Clustering Preview")
    rfm = create_rfm(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=4, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"), use_container_width=True)

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":
    st.title("👥 Customer Segmentation")

    file = st.file_uploader("Upload CSV", type=["csv"])
    data = load_data(file) if file else df

    rfm = create_rfm(data)

    k = st.slider("Clusters", 2, 8, 4)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=k, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"), use_container_width=True)

    st.subheader("Cluster Profile")
    st.dataframe(rfm.groupby("cluster").mean())

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":
    st.title("🎯 Recommendation")

    user_id = st.text_input("Customer ID")
    product_id = st.text_input("Product ID")

    if user_id:
        user_data = df[df["customer_unique_id"] == user_id]

        if user_data.empty:
            st.warning("Cold start → recommend popular")
            rec = df.groupby("product_id")["review_score"].count().sort_values(ascending=False).head(10)
            st.dataframe(rec)
        else:
            profile = user_data.groupby("product_category_name_english")["review_score"].mean()
            prod = df.groupby(["product_id","product_category_name_english"])["review_score"].mean().reset_index()

            merged = prod.merge(profile, on="product_category_name_english", suffixes=("_prod","_user"))
            merged["score"] = merged["review_score_prod"] * 0.7 + merged["review_score_user"] * 0.3

            st.dataframe(merged.sort_values("score", ascending=False).head(10))

    if product_id:
        try:
            category = df[df["product_id"] == product_id]["product_category_name_english"].iloc[0]
            rec = df[df["product_category_name_english"] == category]
            st.dataframe(rec.head(10))
        except:
            st.error("Product not found")

# =========================
# MARKET BASKET
# =========================
elif menu == "🛍️ Market Basket":
    st.title("Market Basket Analysis")

    try:
        rules = pd.read_csv("rules.csv")

        min_lift = st.slider("Min Lift", 0.0, 10.0, 1.0)
        filtered = rules[rules["lift"] >= min_lift]

        st.dataframe(filtered.sort_values("lift", ascending=False).head(20))
        st.plotly_chart(px.scatter(filtered, x="support", y="confidence", size="lift"))

    except:
        st.warning("Run FP-Growth first")

# =========================
# PREDICTION
# =========================
elif menu == "🔮 Prediction":
    st.title("Prediction")

    price = st.number_input("Price", min_value=0.0)
    freight = st.number_input("Freight", min_value=0.0)
    payment = st.number_input("Payment", min_value=0.0)

    if st.button("Predict"):
        try:
            model = joblib.load("classifier.pkl")
            pred = model.predict([[price, freight, payment]])
            st.success(f"Prediction: {pred[0]}")
        except:
            st.error("Train model in Admin tab first")

# =========================
# ADMIN
# =========================
elif menu == "⚙️ Admin":
    st.title("Admin Panel")

    file = st.file_uploader("Upload new dataset", type=["csv"])

    if file:
        new_df = pd.read_csv(file)
        st.success("Uploaded!")
        st.dataframe(new_df.head())

        if st.button("Retrain Model"):
            try:
                X = new_df[["price","freight_value","payment_value"]]
                y = new_df["review_score"]

                X = X.apply(pd.to_numeric, errors="coerce").dropna()
                y = y.loc[X.index]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                pred = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, pred))
                mae = mean_absolute_error(y, pred)

                joblib.dump(model, "classifier.pkl")

                st.success("Model retrained")
                st.write("RMSE:", rmse)
                st.write("MAE:", mae)

            except Exception as e:
                st.error(f"Error: {e}")
