### Amazon Beauty Recommendation System
Project này xây dựng một hệ thống gợi ý sản phẩm làm đẹp (Beauty Products) trên Amazon, sử dụng thuật toán **Matrix Factorization** với kỹ thuật tối ưu **Stochastic Gradient Descent (SGD)**. Điểm đặc biệt của dự án này là toàn bộ quy trình xử lý dữ liệu và thuật toán được xây dựng thủ công bằng thư viện **NumPy**, không sử dụng các framework cấp cao như Scikit-learn hay Pandas cho các tác vụ cốt lõi.

### Mục lục
- [Amazon Beauty Recommendation System](#amazon-beauty-recommendation-system)
- [Mục lục](#mục-lục)
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Method](#method)
- [Installation \& Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges \& Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
  - [Thông tin tác giả](#thông-tin-tác-giả)
  - [Contact](#contact)
- [License](#license)

### Giới thiệu
**Mô tả bài toán**
Trong thời đại thương mại điện tử bùng nổ, người dùng bị choáng ngợp bởi hàng triệu sản phẩm với sợ đa dạng và phong phú. Vấn đề thách thức đối với nền công nghiệp thương mại điện tử là làm sao có thể gợi ý các sản phẩm dựa trên nhu cầu và sở thích của mỗi người dùng. Hệ thống gợi ý (Recommender System) đóng vai trò như một bộ lọc thông minh, giúp dự đoán mức độ yêu thích của người dùng đối với các sản phẩm mà họ chưa từng tương tác dự trên lịch sử tương tác của người dùng đối với các sản phẩm trước đó, từ đó đưa ra các đề xuất phù hợp.

**Động lực \& Ứng dụng**
- **Thực tế:** Amazon có hàng trăm nghìn sản phẩm làm đẹp. Một hệ thống gợi ý tốt giúp tăng trải nghiệm người dùng và doanh thu bán hàng (Cross-selling).
- **Học thuật:** Dự án này nhằm mục đích hiểu sâu về toán học đằng sau các thuật toán **Collaborative Filtering** và kỹ năng tối ưu hóa tính toán trên ma trận thưa (Sparse Matrix) bằng NumPy.

**Mục tiêu cụ thể**
1. Xây dựng pipeline xử lý dữ liệu lớn (1.2M users, 2M ratings) chỉ dùng NumPy.
2. Cài đặt thuật toán **Popularity-based** (Baseline) và **Matrix Factorization**.
3. Đạt chỉ số **RMSE < 1.0** trên tập kiểm thử.

### Dataset
- **Nguồn dữ liệu:** [Amazon Ratings - Beauty Products](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)
- **Kích thước gốc:**
  - **Ratings:** $\sim$ 2,023,070 dòng
  - **Users:** $\sim$ 1,210,271 người
  - **Products:** $\sim$ 249,274 sản phẩm
- **Đặc điểm:**
  - **Features:** `UserId`, `ProductId`, `Rating` (1.0-5.0), `Timestamp` (Unix).
  - **Sparsity (Độ thưa):** > 99.99%. Hầu hết người dùng chỉ đánh giá 1-2 sản phẩm.
  - **Long-tail:** Phân phối tuân theo quy luật 80/20, dữ liệu tập trung vào một số ít sản phẩm nổi tiếng.

### Method
**Quy trình xử lý dữ liệu**
Quy trình được chia làm 2 giai đoạn nối tiếp nhau:

**Giai đoạn 1: Data Ingestion (thực hiện trong Exploration)**
- **Streaming & Caching:** Đọc file CSV theo từng dòng để tối ưu bộ nhớ. Dữ liệu được giữ nguyên định dạng chuỗi (String IDs) và lưu trữ dưới dạng **NumPy Object Array** (`.npy`) để phục vụ cho các bước phân tích và lọc sau này.

**Giai đoạn 2: Data Cleaning & Transformation (thực hiện trong Preprocessing)**
1. **Load Raw Data:** Tải dữ liệu thô (Object Array) từ bộ nhớ đệm (Cache) đã được tạo ở giai đoạn trước.
2. **Iterative K-Core Filtering ($k=5$):** Lọc lặp đi lặp lại trên dữ liệu thô. Chỉ giữ lại các User và Product có ít nhất 5 tương tác để loại bỏ nhiễu. Bước này giúp giảm độ thưa và loại bỏ nhiễu (Noise Reduction).
3. **Map Construction & Conversion:**
    - Sau khi lọc sạch, hệ thống mới tiến hành tạo Dictionary Mapping (`String ID` $\leftrightarrow$ `Int Index`).
    - Chuyển đổi toàn bộ dữ liệu sang dạng số (`float32`).
4. **Feature Engineering:** Tạo đặc trưng `Time_Weight` (Trọng số thời gian) và `Year`.
5. **Statistical Hypothesis Testing:** Kiểm định Z-Test để xác nhận xu hướng tăng trưởng rating.
6. **Standardization:** Chuẩn hóa Z-Score cho cột Rating.
7. **Time-based Split:** Chia Train/Test theo thời gian.

**Thuật toán (Models)**
Được cài đặt trong `src/models.py`:

**Baseline: Popularity Recommender**
- Gợi ý dựa trên điểm trung bình của sản phẩm (Mean Rating).
- Sử dụng kỹ thuật Vectorization (`np.unique`, `np.add.at`) để tính toán hiệu quả.

**Advanced: Matrix Factorization (SGD)**
Mô hình phân rã ma trận Rating ($R$) thành hai ma trận ẩn $P$ (User) và $Q$ (Product).
- **Công thức dự đoán:**
  $$
  \hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i^T
  $$
- **Hàm mất mát (Weighted MSE):**
  $$
  L = \sum (r_{ui} - \hat{r}_{ui})^2 \times W_{ui} + \lambda(||\theta||^2)
  $$
- **Tối ưu hóa:** Cài đặt thủ công vòng lặp**Stochastic Gradient Descent (SGD)**. Trong mỗi epoch, dữ liệu được xáo trộn (`shuffle`) và cập nhật trọng số từng mẫu một.
### Installation & Setup
Dự án yêu cầu Python phiên bản 3.8 trở lên. Khuyến khích sử dụng môi trường ảo để tránh xung đột thư viện.

**Bước 1: Clone repository**
Tải mã nguồn dự án về máy của bạn:
```bash
git clone https://github.com/TruongQuangPhat/amazon-beauty-recsys.git
cd amazon-beauty-recsys
```
**Bước 2: Tạo và Kích hoạt môi trường ảo (Virtual Environment)**
Việc này giúp cô lập các thư viện của dự án với hệ thống chính.

**Trên Windowns:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Trên macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt thư viện phụ thuộc**
Cài đặt các thư viện cần thiết (NumPy, Matplotlib, Seaborn) được liệt kê trong file `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Bước 4: Chuẩn bị dữ liệu**
1. Truy cập vào Kaggle để tải bộ dữ liệu: [Amazon Ratings - Beauty Products](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings).
2. Giải nén `ratings_Beauty.csv`.
3. Đặt file dữ liệu vào đường dẫn thư mục sau: `data/raw/ratings_Beauty.csv`.

### Usage
Chạy lần lượt các Notebook theo thứ tự:

**Bước 1: Khám phá \& Nạp dữ liệu (Exploration)**
- Chạy `notebooks/01_data_exploration.ipynb`.
- **Nhiệm vụ:** Đọc file CSV gốc, chuyển đổi sang NumPy Object Array và lưu file tạm (`data_raw.npy`) vào thư mục `processed/`.
- **Phân tích:** Vẽ các biểu đồ phân phối và xu hướng để hiểu dữ liệu.
- **Câu hỏi có ý nghĩa:** Hiểu sâu hơn về dữ liệu thông qua việc trả lời các câu hỏi có ý nghĩa.

**Bước 2: Tiền xử lý (Preprocessing)**
- Chạy `notebooks/02_preprocessing.ipynb`.
- **Nhiệm vụ:** Thực thi pipeline Load Raw $\to$ Filter K-Core $\to$ Build Maps $\to$ Add Features $\to$ Standardize $\to$ Split.
- **Output:** Lưu các file thành phẩm (`train.npy`, `test.npy`, `user_map.pkl`, `product_map.pkl`) vào `data/processed/` để sẵn sàng cho Training.

**Bước 3: Huấn luyện \& Đánh giá (Modeling)**
- Chạy `notebooks/03_modeling.ipynb`.
- Huấn luyện **Popularity Model** và **Matrix Factorization**.
- So sánh RMSE và vẽ biểu đồ Learning Curve.
- **Chạy Demo:** Random một UserId trong tập test để hiển thị lịch sử tương tác của user và danh sách các sản phẩm gợi ý cụ thể.

### Results
**Metrics (RMSE)**
Đánh giá trên tập Test (20% dữ liệu tương lai):
|Model|RMSE|Nhận xét|
|:----|:--:|:-------|
|**Popularity (Baseline)**|**1.0085**|Mức chuẩn khá tốt do hiệu ứng đám đông.|
|**Matrix Factorization**|**0.9564**|Cải thiện ~5.16%. Sai số dự đoán < 1 sao|

**Phân tích**
- MF Model vượt qua Baseline, chứng tỏ khả năng học được các sở thích cá nhân hóa.
- Biểu đồ Learning Curve cho thấy mô hình hội tụ ổn định, không bị Overfitting quá mức nhờ Regularization.
- Việc sử dụng **Time Decay** giúp mô hình thích nghi tốt với sự bùng nổ dữ liệu vào năm 2014.
### Project Structure
```plaintext
amazon-beauty-recsys/
├── data/
│   ├── raw/                          # Chứa file CSV gốc
│   └── processed/                    # Chứa file .npy (Train/Test) và .pkl (Maps)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA trên dữ liệu thô
│   ├── 02_preprocessing.ipynb        # Pipeline làm sạch và biến đổi dữ liệu
│   └── 03_modeling.ipynb             # Huấn luyện và Đánh giá mô hình
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Chứa các hàm xử lý dữ liệu cốt lõi
│   ├── visualization.py              # Chứa các hàm vẽ biểu đồ
│   └── models.py                     # Chứa class Popularity và MatrixFactorization
└── README.md
```

### Challenges & Solutions
Trong quá trình thực hiện project với ràng buộc chỉ sử dụng Numpy, một số khó khăn đã gặp phải như sau:
1. **Quy trình xử lý dữ liệu lớn**
    - **Thách thức:** Việc đọc đi đọc lại file CSV 2GB tốn nhiều thời gian và tài nguyên.
    - **Giải pháp:** Tách biệt bước **Ingestion** (trong Explore) và **Processing** (trong Preprocessing). Sử dụng file `.npy` làm cầu nối trung gian giúp tốc độ load dữ liệu ở các bước sau nhanh hơn gấp nhiều lần.
2. **Xử lý dữ liệu hỗn hợp**
    - **Thách thức:** NumPy thuần không hỗ trợ tốt cột String chung với số.
    - **Giải pháp:** Sử dụng `dtype=object` ở giai đoạn đầu để giữ ID gốc, sau đó áp dụng kỹ thuật **Late Mapping** để chuyển sang `float32` ngay sau khi lọc rác xong.
3. **Vấn đề hiệu năng khi xử lý chuỗi**
    - **Thách thức:** Dữ liệu thô chứa ID dạng chuỗi (String), buộc phải sử dụng `dtype=object` để lưu trữ. Tuy nhiên, các phép toán tập hợp như `np.unique` hay `np.isin` trên mảng Object rất chậm (do NumPy phải so sánh từng chuỗi ký tự thay vì so sánh số, mất đi khả năng tối ưu hóa cấp thấp của C).
    - **Giáp pháp:** Chuyển đổi các kiểu dữ liệu Object sang dạng số để có thể thực hiện các phép toán `np.unique` và `np.isin` một cách hiệu quả hơn.
4. **Vấn đề Lỗ hổng Index và Bộ nhớ**
    - **Thách thức:** Trong các phiên bản thử nghiệm đầu tiên, quy trình lọc K-Core để lại các chỉ số không liên tục. Ví dụ: Dù chỉ còn 100 User, nhưng User ID vẫn có giá trị lên tới 1.000.000. Điều này khiến mô hình Matrix Factorization phải khởi tạo ma trận khổng lồ kích thước `(1.000.000 x K)` thay vì `(100 x K)`, gây lãng phí RAM nghiêm trọng và làm chậm quá trình huấn luyện do phải xử lý hàng triệu dòng rỗng.
    - **Giải pháp:** Thực hiện convert kiểu dữ liệu Object của các dữ liệu ID sang kiểu dữ liệu Int sang khi lọc K-Core để đảm bảo các chỉ số liên tục mà không xảy ra trình trạng Index Gaps. Điều này giúp cho việc huấn luyện mô hình hiệu quả hơn.

### Future Improvements
- **Tối ưu hóa Hyperparameter:** Sử dụng Grid Search để tìm ra bộ tham số (Learning rate, Regularization, Factors) tốt nhất.
- **Hybrid System:** Kết hợp thêm thông tin nội dung (Content-based) để giải quyết triệt để vấn đề Cold Start cho sản phẩm mới.
- **Cập nhật Dữ liệu & Học liên tục:** Do đặc thù ngành làm đẹp thay đổi xu hướng rất nhanh, việc sử dụng dữ liệu cũ (2014) có thể không phản ánh đúng thị hiếu hiện tại. Hướng phát triển tiếp theo là thiết lập luồng dữ liệu (Data Pipeline) để cập nhật các đánh giá mới nhất (2024-2025), giúp mô hình thích nghi với các xu hướng trong tương lai gần.
### Contributors
#### Thông tin tác giả
- **Họ tên:** Trương Quang Phát
- **MSSV:** 23120318
- **Lớp:** CQ2023/21
- **Trường:** Đại học Khoa học Tự nhiên, ĐHQG TP.HCM (HCMUS)
#### Contact
- **Email:** 23120318@student.hcmus.edu.vn
### License
Project này được thực hiện cho mục đích học tập trong môn học **Lập trình cho Khoa học Dữ liệu**.