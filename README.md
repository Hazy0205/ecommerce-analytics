# 🛒 Hệ thống Phân tích & Gợi ý Sản phẩm Thương mại Điện tử

## 📌 Tổng quan

Dự án xây dựng một hệ thống phân tích dữ liệu và gợi ý sản phẩm cho nền tảng thương mại điện tử dựa trên bộ dữ liệu Olist (Brazil). Mục tiêu của bài toán là khai thác dữ liệu để hiểu hành vi khách hàng, dự đoán đánh giá sản phẩm, phân khúc khách hàng và xây dựng hệ thống gợi ý nhằm nâng cao trải nghiệm người dùng.

Hệ thống được triển khai dưới dạng ứng dụng web tương tác sử dụng Streamlit, giúp người dùng dễ dàng khám phá dữ liệu và các kết quả phân tích.

---

## 📊 Dataset

* Nguồn: Olist E-commerce Dataset (Brazil)
* Dữ liệu đã xử lý:

  * `cleaned_data_small.csv`
  * `rfm_data.csv`
  * `rules.csv`

Dữ liệu bao gồm thông tin đơn hàng, khách hàng, sản phẩm, thanh toán và đánh giá.

---

## ⚙️ Phương pháp thực hiện

### 🔹 1. Xử lý dữ liệu & Feature Engineering

* Làm sạch dữ liệu, xử lý missing values
* Kết hợp nhiều bảng dữ liệu
* Tạo đặc trưng RFM:

  * Recency
  * Frequency
  * Monetary

---

### 🔹 2. Phân loại (Classification)

Sử dụng nhiều mô hình:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine
* Naive Bayes

Đánh giá bằng:

* Accuracy
* F1-score

---

### 🔹 3. Dự đoán (Regression)

* Random Forest Regressor được sử dụng để dự đoán review score
* Input:

  * price
  * freight_value
  * payment_value

---

### 🔹 4. Phân cụm khách hàng (Clustering)

* Thuật toán: KMeans
* Dữ liệu: RFM

Mục tiêu:

* Xác định nhóm khách hàng giá trị cao
* Phát hiện nhóm khách hàng có nguy cơ rời bỏ

---

### 🔹 5. Hệ thống gợi ý (Recommendation)

* Dựa trên lịch sử mua hàng của khách
* Phân biệt:

  * User cũ → gợi ý cá nhân hóa
  * User mới → gợi ý phổ biến

---

### 🔹 6. Market Basket Analysis

* Thuật toán: FP-Growth
* Sinh các luật kết hợp giữa các sản phẩm

Chỉ số:

* Support
* Confidence
* Lift

---

## 🖥️ Ứng dụng Streamlit

Ứng dụng bao gồm các chức năng chính:

* 📊 Dashboard: Tổng quan dữ liệu và doanh thu
* 👥 Segmentation: Phân cụm khách hàng
* 🎯 Recommendation: Gợi ý sản phẩm
* 🛍️ Market Basket: Luật kết hợp
* 🔮 Prediction: Dự đoán review score
* ⚙️ Admin: Upload dữ liệu và train model

---

## 📊 Model Performance & Key Insights

Kết quả thực nghiệm cho thấy các mô hình học máy đạt hiệu năng ở mức trung bình do đặc tính dữ liệu thương mại điện tử thường khá phân tán và không cân bằng. Trong đó, Random Forest và Gradient Boosting là hai mô hình cho kết quả tốt nhất với Accuracy và F1-score cao hơn so với các mô hình còn lại.

Phân tích RFM cho thấy sự phân hóa rõ rệt giữa các nhóm khách hàng. Một số nhóm có giá trị chi tiêu cao nhưng thời gian mua gần nhất xa, cho thấy nguy cơ rời bỏ, cần được ưu tiên giữ chân. Ngược lại, các nhóm có tần suất mua cao thể hiện mức độ trung thành tốt.

Trong Market Basket Analysis, nhiều sản phẩm thuộc cùng danh mục thường được mua cùng nhau, tạo cơ hội triển khai chiến lược cross-selling và bundle sản phẩm.

Hệ thống recommendation cho thấy hành vi mua sắm mang tính cá nhân hóa cao, giúp nâng cao trải nghiệm người dùng và tăng khả năng chuyển đổi.

Ngoài ra, mô hình dự đoán review score cho thấy có mối liên hệ giữa giá trị đơn hàng và mức độ hài lòng của khách hàng.

---

## 📈 Model Performance

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.56     | 0.41     |
| Random Forest       | 0.57     | 0.53     |
| Gradient Boosting   | 0.56     | 0.53     |
| SVM                 | 0.54     | 0.44     |
| Naive Bayes         | 0.42     | 0.42     |

---

## 📸 Demo

*(Thêm ảnh giao diện Streamlit tại đây để tăng điểm presentation)*

---

## 🚀 Cách chạy

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Cấu trúc project

```
├── app.py
├── cleaned_data_small.csv
├── rfm_data.csv
├── rules.csv
├── classifier.pkl
├── requirements.txt
└── README.md
```

---

## 🧠 Công nghệ sử dụng

* Python (Pandas, NumPy)
* Scikit-learn
* Plotly
* Streamlit
* Joblib
* MLxtend

---

## 📌 Kết luận

Dự án chứng minh rằng việc kết hợp phân tích dữ liệu và machine learning có thể xây dựng hệ thống gợi ý thông minh, giúp cải thiện trải nghiệm người dùng và hỗ trợ ra quyết định trong thương mại điện tử.

---

## 👨‍💻 Thông tin

* Họ tên: NHÓM 10
* Môn học: Big Data
* Trường: HCMUTE
