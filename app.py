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
# RECOMMENDATION (SURPRISE)
# =========================
elif menu == "🎯 Recommendation":
    st.subheader("Product Recommendation (SVD)")

    data = df[["customer_unique_id","product_id","review_score"]].dropna()

    reader = Reader(rating_scale=(1,5))
    dataset = Dataset.load_from_df(data, reader)

    trainset = dataset.build_full_trainset()
    model = SVD()
    model.fit(trainset)

    user_id = st.text_input("Enter Customer ID")

    if user_id:
        all_products = df["product_id"].unique()

        # 🔥 CHECK USER EXIST
        known_users = data["customer_unique_id"].unique()

        if user_id not in known_users:
            st.warning("User mới → recommend theo sản phẩm phổ biến")

            popular = (
                df.groupby("product_id")["review_score"]
                .count()
                .sort_values(ascending=False)
                .head(10)
                .index
            )

            rec_df = pd.DataFrame(popular, columns=["product_id"])
            st.write(rec_df)

        else:
            # 🔥 LOẠI SẢN PHẨM ĐÃ MUA
            bought = df[df["customer_unique_id"] == user_id]["product_id"].unique()

            candidates = [p for p in all_products if p not in bought]

            preds = []

            for p in candidates[:300]:
                pred = model.predict(user_id, p)
                preds.append((p, pred.est))

            preds = sorted(preds, key=lambda x: x[1], reverse=True)[:10]

            rec_df = pd.DataFrame(preds, columns=["product_id","score"])
            st.write(rec_df)

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
            st.success(f"Predicted Score: {pred[0]}")
        except:
            st.error("Train model first in Admin tab")

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
        # 🔥 đảm bảo X và y cùng index
        data_model = df[["price","freight_value","payment_value","review_score"]].dropna()

        X = data_model[["price","freight_value","payment_value"]]
        y = data_model["review_score"]

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        joblib.dump(model, "classifier.pkl")

        st.success("Model retrained!")
