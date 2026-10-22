# RESEARCH BLUEPRINT 05

## Limit Order Book & Liquidity Stress Prediction

*Dự báo biến động giá ngắn hạn và stress thanh khoản từ Limit Order Book của thị trường crypto*

> **Môn học gợi ý:** Công nghệ ứng dụng trong Tài chính / Financial Technology  
> **Định dạng:** Research + Code + Financial Product  
> **Mức độ:** Undergraduate advanced / MSc-entry level  
> **Thời lượng phù hợp:** 8–12 tuần  
> **Ngôn ngữ triển khai:** Python ưu tiên; R có thể dùng cho econometrics  
> **Lưu ý:** Đây là thiết kế học thuật và mô phỏng nghiên cứu, không phải khuyến nghị đầu tư.

---

## 1. Tên chủ đề

**Limit Order Book and Liquidity Stress Prediction in Cryptocurrency Markets: From Order Flow Imbalance to DeepLOB and Transformers**

**Hướng nghiên cứu:** High-Frequency Finance, Market Microstructure, Deep Learning, Liquidity Risk.

---

## 2. Mục tiêu nghiên cứu, câu hỏi nghiên cứu và mô hình dự kiến

### 2.1. Mục tiêu nghiên cứu

Dự án khai thác Limit Order Book (LOB) để nghiên cứu thông tin chứa trong bid/ask depth trước những chuyển động cực ngắn của mid-price và liquidity. Khác với dự báo daily crypto returns, đây là bài toán microstructure có cấu trúc rõ, yêu cầu xử lý dữ liệu lớn và đánh giá theo time-order.

Mục tiêu:

1. Xây pipeline thu thập/đọc LOB.
2. Xây order-flow và liquidity features.
3. Dự báo direction của mid-price ở horizon ngắn.
4. Dự báo liquidity stress/spread widening.
5. So sánh linear model, tree model và DeepLOB/Transformer.

### 2.2. Câu hỏi nghiên cứu

- **RQ1:** Order-book imbalance có dự báo mid-price movements hay không?
- **RQ2:** Order Flow Imbalance (OFI) có nhiều thông tin hơn static depth imbalance không?
- **RQ3:** DeepLOB/Transformer cải thiện bao nhiêu so với logistic/XGBoost?
- **RQ4:** Model có ổn định khi volatility regime thay đổi không?
- **RQ5:** Accuracy có còn ý nghĩa sau khi đưa uncertainty và class imbalance vào đánh giá không?

### 2.3. Mid-price và spread

\[
m_t=\frac{a_t^1+b_t^1}{2}
\]

\[
Spread_t=a_t^1-b_t^1
\]

với \(a_t^1,b_t^1\) là best ask và best bid.

### 2.4. Order-book imbalance

Một measure đơn giản:

\[
OBI_t=
\frac{\sum_{l=1}^{L}Q^{bid}_{l,t}
-\sum_{l=1}^{L}Q^{ask}_{l,t}}
{\sum_{l=1}^{L}Q^{bid}_{l,t}
+\sum_{l=1}^{L}Q^{ask}_{l,t}}
\]

Nên kiểm tra \(L=1,5,10,20\).

### 2.5. Prediction target

Mid-price direction:

\[
y_{t,h}=
\begin{cases}
1,& m_{t+h}-m_t>\delta\\
0,& |m_{t+h}-m_t|\le\delta\\
-1,& m_{t+h}-m_t<-\delta
\end{cases}
\]

Horizon có thể tính theo:

- 10/50/100 events;
- 1/5/10 seconds.

Chọn một thiết kế nhất quán.

### 2.6. Baseline models

- Logistic Regression;
- Multinomial Logistic;
- Random Forest;
- XGBoost.

### 2.7. Deep-learning models

**DeepLOB-style architecture**
- convolutional layers xử lý spatial book structure;
- LSTM xử lý time dynamics.

**Temporal CNN** hoặc **Transformer** là model nâng cao.

Không chạy DL chỉ vì “hiện đại”. Nhóm phải chứng minh incremental performance trên untouched test period.

### 2.8. Liquidity-stress model

Định nghĩa stress khi spread hoặc price impact vượt percentile:

\[
Stress_{t+h}
=
\mathbf{1}(Spread_{t+h}>Q_{0.95})
\]

hoặc dùng multi-dimensional label gồm spread, depth depletion và short-term volatility.

Mô hình:

\[
P(Stress_{t+h}=1|LOB_t,OFI_t,Trades_t)
\]

### 2.9. Validation

Bắt buộc walk-forward.

Metrics:

- macro F1;
- balanced accuracy;
- MCC;
- precision/recall;
- Brier score nếu output probability;
- calibration curve.

Không random shuffle train/test.

---

## 3. Data dự kiến

### 3.1. Markets

Ưu tiên:

- BTC/USDT;
- ETH/USDT.

### 3.2. Nguồn

- Binance websocket/order-book streams;
- Binance historical trade data;
- Coinbase nếu nhóm muốn cross-exchange validation.

### 3.3. Raw fields

Mỗi snapshot:

```text
timestamp
bid_price_1 ... bid_price_L
bid_qty_1   ... bid_qty_L
ask_price_1 ... ask_price_L
ask_qty_1   ... ask_qty_L
```

Trades:

```text
trade_timestamp
price
quantity
aggressor_side
```

### 3.4. Feature engineering

- spread;
- microprice;
- OBI;
- OFI;
- cumulative depth;
- depth slope;
- trade imbalance;
- cancellation/addition intensities nếu có event feed;
- short-term realized volatility.

### 3.5. Storage

Do dữ liệu lớn, khuyến nghị:

- Parquet;
- partition theo date/symbol;
- Polars/Pandas;
- DuckDB cho exploratory queries.

---

## 4. Output dự kiến

### 4.1. Academic output

Báo cáo cần trình bày:

- market microstructure;
- LOB representation;
- target definition;
- leakage controls;
- model comparison;
- regime robustness.

Bảng:

1. class distribution;
2. feature summary;
3. baseline metrics;
4. deep model metrics;
5. performance by volatility regime;
6. latency/model complexity.

Hình:

- LOB heatmap;
- OBI vs future returns;
- confusion matrices;
- calibration;
- model performance through time.

### 4.2. Product output

**Liquidity Stress Meter**

Hiển thị:

```text
BTC/USDT
Spread: ...
Depth imbalance: ...
OFI: ...
P(up): ...
P(flat): ...
P(down): ...
Liquidity-stress probability: ...
```

Có thể replay historical order book thay vì live trading.

### 4.3. Giá trị đào tạo

Đây là dự án tốt cho sinh viên học:

- streaming data;
- high-frequency finance;
- deep learning;
- reproducibility;
- memory-efficient computing.

---

## 5. Tài liệu tham khảo gợi ý

1. Chordia, T., Roll, R., & Subrahmanyam, A. (2002). Order imbalance, liquidity, and market returns. *Journal of Financial Economics, 65*(1), 111–130.
2. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing, 67*(11), 3001–3012.
3. Jha, R., De Paepe, M., Holt, S., West, J., & Ng, S. (2020). *Deep Learning for Digital Asset Limit Order Books*. SSRN/working paper.
4. Schnaubelt, M. (2022). Deep reinforcement learning for the optimal placement of cryptocurrency limit orders. *European Journal of Operational Research, 296*(3), 993–1006.
5. Almeida, J., & Gonçalves, T. C. (2024). Cryptocurrency market microstructure: A systematic literature review. *Annals of Operations Research, 332*, 1035–1068.
6. Lucchese, L., Pakkanen, M. S., & Veraart, A. E. D. (2024). The short-term predictability of returns in order book markets: A deep learning perspective. *International Journal of Forecasting, 40*(4), 1587–1621.
7. *Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data* (2026). *Decisions in Economics and Finance*.


---

## 6. Checklist bàn giao tối thiểu cho nhóm

- [ ] Research question được phát biểu rõ và nhất quán với model.
- [ ] Data dictionary và source log.
- [ ] Script tải/chuẩn hóa dữ liệu có thể chạy lại.
- [ ] Không có look-ahead bias/data leakage.
- [ ] Có econometric/statistical benchmark.
- [ ] Có ít nhất một model nâng cao.
- [ ] Có untouched out-of-sample test.
- [ ] Có robustness hoặc sensitivity analysis.
- [ ] Có README hướng dẫn chạy code.
- [ ] Có dashboard/app hoặc research visualization có thể demo.
- [ ] Có báo cáo học thuật giải thích economic meaning, không chỉ model metrics.

## 7. Cấu trúc repository đề xuất

```text
project/
├── README.md
├── requirements.txt
├── config/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── notebooks/
├── dashboard/
├── tests/
├── outputs/
│   ├── tables/
│   └── figures/
└── paper/
```

> **Nguyên tắc chung:** một model đơn giản nhưng point-in-time đúng, có benchmark và kiểm định out-of-sample tốt hơn một model phức tạp nhưng leakage hoặc không thể tái lập.
