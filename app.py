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
# STYLE (MULTI COLOR UI)
# =========================
st.markdown("""
<style>
.block-dashboard {background-color:#E3F2FD;padding:20px;border-radius:10px;}
.block-segmentation {background-color:#E8F5E9;padding:20px;border-radius:10px;}
.block-recommendation {background-color:#FFF3E0;padding:20px;border-radius:10px;}
.block-market {background-color:#FCE4EC;padding:20px;border-radius:10px;}
.block-prediction {background-color:#F3E5F5;padding:20px;border-radius:10px;}
.block-admin {background-color:#ECEFF1;padding:20px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
def load_data(file=None):
    if file:
        return pd.read_csv(file)
    return pd.read_csv("cleaned_data_small.csv")

# =========================
# RFM FUNCTION
# =========================
def create_rfm(data):
    rfm = data.groupby("customer_unique_id").agg({
        "order_purchase_timestamp": "max",
        "order_id": "count",
        "payment_value": "sum"
    }).reset_index()

    rfm.columns = ["customer_id","Recency","Frequency","Monetary"]

    rfm["Recency"] = (pd.to_datetime("today") - pd.to_datetime(rfm["Recency"], errors='coerce')).dt.days

    for col in ["Recency","Frequency","Monetary"]:
        rfm[col] = pd.to_numeric(rfm[col], errors="coerce")

    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
    return rfm

# =========================
# MENU
# =========================
menu = st.sidebar.radio("Menu", [
    "📊 Dashboard",
    "👥 Segmentation",
    "🎯 Recommendation",
    "🛍️ Market Basket",
    "🔮 Prediction",
    "⚙️ Admin",
])

df = load_data()

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":
    st.markdown('<div class="block-dashboard">', unsafe_allow_html=True)
    st.title("📊 Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Orders", df["order_id"].nunique())
    col2.metric("Customers", df["customer_unique_id"].nunique())
    col3.metric("Revenue", f"${df['payment_value'].sum():,.0f}")

    top_cat = df.groupby("product_category_name_english")["payment_value"].sum().head(10)
    st.plotly_chart(px.bar(top_cat, title="Top Categories"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# SEGMENTATION
# =========================
elif menu == "👥 Segmentation":
    st.markdown('<div class="block-segmentation">', unsafe_allow_html=True)
    st.title("👥 Segmentation")

    file = st.file_uploader("Upload CSV", type=["csv"])
    data = load_data(file) if file else df

    if set(["Recency","Frequency","Monetary"]).issubset(data.columns):
        rfm = data.copy()
    else:
        rfm = create_rfm(data)

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])

    model = KMeans(n_clusters=4, random_state=42)
    rfm["cluster"] = model.fit_predict(X)

    st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="cluster"))

    st.dataframe(rfm.groupby("cluster")[["Recency","Frequency","Monetary"]].mean())

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RECOMMENDATION
# =========================
elif menu == "🎯 Recommendation":
    st.markdown('<div class="block-recommendation">', unsafe_allow_html=True)
    st.title("🎯 Recommendation")

    user_id = st.text_input("Customer ID")

    if user_id:
        rec = df.groupby("product_id")["review_score"].count().sort_values(ascending=False).head(10)
        st.dataframe(rec)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MARKET BASKET
# =========================
elif menu == "🛍️ Market Basket":
    st.markdown('<div class="block-market">', unsafe_allow_html=True)
    st.title("Market Basket")

    try:
        rules = pd.read_csv("rules.csv")
        st.dataframe(rules.head(20))
    except:
        st.warning("No rules file")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
elif menu == "🔮 Prediction":
    st.markdown('<div class="block-prediction">', unsafe_allow_html=True)
    st.title("Prediction")

    price = st.number_input("Price", 0.0)
    freight = st.number_input("Freight", 0.0)
    payment = st.number_input("Payment", 0.0)

    if st.button("Predict"):
        model = joblib.load("classifier.pkl")
        pred = model.predict([[price, freight, payment]])
        st.success(pred)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ADMIN
# =========================
elif menu == "⚙️ Admin":
    st.markdown('<div class="block-admin">', unsafe_allow_html=True)
    st.title("Admin")

    file = st.file_uploader("Upload data", type=["csv"])

    if file:
        new_df = pd.read_csv(file)

        if st.button("Retrain Model"):
            X = new_df[["price","freight_value","payment_value"]].apply(pd.to_numeric, errors="coerce")
            y = pd.to_numeric(new_df["review_score"], errors="coerce")

            data_clean = pd.concat([X, y], axis=1).dropna()
            X = data_clean[["price","freight_value","payment_value"]]
            y = data_clean["review_score"]

            model = RandomForestRegressor()
            model.fit(X, y)

            joblib.dump(model, "classifier.pkl")

            st.success("Done")

    st.markdown('</div>', unsafe_allow_html=True)
