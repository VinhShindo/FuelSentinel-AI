# FuelSentinel-AI: Time Series Event Detection for Fuel Monitoring

## 1. Tổng quan hệ thống

FuelSentinel-AI là hệ thống giám sát mức nhiên liệu theo thời gian thực nhằm phát hiện các sự kiện bất thường trên phương tiện như:

- Driving
- Idle
- Refuel
- Fuel Theft

Hệ thống nhận dữ liệu liên tục từ cảm biến nhiên liệu và GPS, thực hiện xử lý tín hiệu, trích xuất đặc trưng, sau đó sử dụng mô hình Machine Learning để phân loại trạng thái hoạt động của xe.

Khác với các bài toán phân loại ảnh truyền thống, đây là bài toán **Time Series Event Detection**, trong đó mỗi quyết định được đưa ra dựa trên một chuỗi các mẫu dữ liệu theo thời gian thay vì một mẫu độc lập.

---

# 2. Kiến trúc tổng thể

```
                 Fuel Sensor + GPS
                        │
                        ▼
             Data Acquisition Layer
                        │
                        ▼
           Calibration (ADC → Liter)
                        │
                        ▼
          Data Cleaning & Synchronization
                        │
                        ▼
            Feature Engineering Layer
                        │
                        ▼
      Sliding Window + Linear Regression
            (Regression Slope)
                        │
                        ▼
             Feature Vector Generation
                        │
                        ▼
        Machine Learning Classifier
                        │
                        ▼
 Driving │ Idle │ Refuel │ Fuel Theft
                        │
                        ▼
 Dashboard │ REST API │ Excel Report
```

Linear Regression **không phải mô hình phân loại**, mà chỉ là một bước trong quá trình Feature Engineering nhằm tính toán xu hướng thay đổi của mức nhiên liệu (Regression Slope).

---

# 3. Bộ dữ liệu đầu vào (Raw Sensor Dataset)

Đây là dữ liệu thô được truyền từ thiết bị IoT trước khi trải qua bất kỳ bước xử lý nào.

| STT | Timestamp | VehicleID | Latitude | Longitude | Heading (°) | Speed (km/h) | ADC | Fuel (L) |
|-----|-----------|-----------|----------|-----------|-------------|--------------|-----|----------|
| 1 | 08:00:00 | VN001 | 21.0285 | 105.8541 | 92 | 45 | 1865 | 50.2 |
| 2 | 08:00:05 | VN001 | 21.0287 | 105.8542 | 93 | 43 | 1863 | 50.1 |
| 3 | 08:00:10 | VN001 | 21.0289 | 105.8545 | 94 | 40 | 1860 | 50.0 |
| ... | | | | | | | | |

Trong đó:

| Feature   | Ý nghĩa                                |
| --------- | -------------------------------------- |
| Timestamp | Thời gian lấy mẫu                      |
| VehicleID | Mã phương tiện                         |
| Latitude  | Vĩ độ GPS                              |
| Longitude | Kinh độ GPS                            |
| Heading   | Hướng di chuyển của xe (0–360°)        |
| Speed     | Vận tốc xe (km/h)                      |
| ADC       | Giá trị ADC đọc từ cảm biến nhiên liệu |
| Fuel      | Mức nhiên liệu sau Calibration (Lít)   |

**Ghi chú về Heading**: Trường này tuy chưa được sử dụng trong mô hình hiện tại nhưng sẽ rất hữu ích khi mở rộng bài toán:
- Phát hiện xe quay đầu.
- Phân tích hành vi lái xe.
- Kết hợp GPS để phát hiện gian lận.
- Theo dõi quỹ đạo phương tiện.

---

# 4. Các kịch bản mô phỏng

Để huấn luyện và đánh giá mô hình, dữ liệu được mô phỏng theo các chuỗi hành vi thực tế của phương tiện thay vì sinh ngẫu nhiên hoàn toàn. Mỗi kịch bản kéo dài từ vài chục giây đến vài phút.

Dưới đây là các bảng dữ liệu mẫu mô tả đặc trưng cho từng hành vi trong chuỗi thời gian (với giả định chu kỳ lấy mẫu là 5 giây):

---

## 4.1. Kịch bản Driving

**Đặc điểm:** Speed > 0, Fuel giảm từ từ, GPS và Heading thay đổi liên tục.

| Thời gian | Tốc độ (km/h) | Nhiên liệu (L) | Trạng thái / Đặc điểm |
| :--- | :---: | :---: | :--- |
| 00:00:00 | 45 | 50.2 | Bắt đầu hành trình |
| 00:00:05 | 42 | 50.0 | Xe đang di chuyển |
| 00:00:10 | 40 | 49.8 | Xe đang di chuyển |
| 00:00:15 | 38 | 49.6 | Giảm tốc để rẽ |
| 00:00:20 | 35 | 49.4 | Xe đang di chuyển |
| ... | ... | ... | ... |

---

## 4.2. Kịch bản Idle

**Đặc điểm:** Speed = 0, Fuel gần như không đổi, GPS và Heading đứng yên.

| Thời gian | Tốc độ (km/h) | Nhiên liệu (L) | Trạng thái / Đặc điểm |
| :--- | :---: | :---: | :--- |
| 00:01:00 | 0 | 48.5 | Dừng đèn đỏ |
| 00:01:05 | 0 | 48.5 | Nhiên liệu ổn định |
| 00:01:10 | 0 | 48.5 | Chờ tải hàng |
| 00:01:15 | 0 | 48.4 | Khởi động máy nhưng chưa đi |
| 00:01:20 | 0 | 48.4 | Vẫn đứng yên (Idle) |

---

## 4.3. Kịch bản Refuel

**Đặc điểm:** Speed = 0, Fuel tăng nhanh (đột ngột), GPS không đổi, sau khi nạp xong xuất hiện trạng thái Plateau.

| Thời gian | Tốc độ (km/h) | Nhiên liệu (L) | Trạng thái / Đặc điểm |
| :--- | :---: | :---: | :--- |
| 00:02:00 | 0 | 15.0 | Bắt đầu đổ xăng |
| 00:02:05 | 0 | 20.5 | Nhiên liệu tăng nhanh |
| 00:02:10 | 0 | 28.0 | Nhiên liệu tăng nhanh |
| 00:02:15 | 0 | 35.5 | Nhiên liệu tăng nhanh |
| 00:02:20 | 0 | 46.0 | Kết thúc đổ xăng (Plateau) |
| 00:02:25 | 0 | 46.0 | Ổn định sau đổ xăng |

---

## 4.4. Kịch bản Fuel Theft

**Đặc điểm:** Speed = 0, Fuel giảm rất nhanh, GPS không đổi.

| Thời gian | Tốc độ (km/h) | Nhiên liệu (L) | Trạng thái / Đặc điểm |
| :--- | :---: | :---: | :--- |
| 00:03:00 | 0 | 46.0 | Đang đỗ xe |
| 00:03:05 | 0 | 42.5 | Rút nhiên liệu |
| 00:03:10 | 0 | 38.0 | Rút nhiên liệu |
| 00:03:15 | 0 | 34.5 | Rút nhiên liệu |
| 00:03:20 | 0 | 30.0 | Rút nhiên liệu |
| 00:03:25 | 0 | 30.0 | Dừng rút (Đã bị mất cắp 16L) |

---

## 4.5. Kịch bản tổng hợp (Pipeline thực tế)

Đây là chuỗi trạng thái chính được sử dụng để sinh dữ liệu mô phỏng huấn luyện trong hệ thống, mô phỏng một hành trình hoạt động hoàn chỉnh của xe:

```
Driving
    ↓
  Idle
    ↓
  Refuel
    ↓
 Driving
    ↓
  Idle
    ↓
Fuel Theft
    ↓
 Driving
    ↓
  Idle
```

---

# 5. Calibration

Giá trị ADC không phản ánh trực tiếp lượng nhiên liệu.

Do đó cần chuyển đổi sang đơn vị Lít.

Ví dụ:
```
ADC = 1865 → 50.2 Liter
```

Có thể sử dụng:
- Linear Interpolation
- Polynomial Regression
- Lookup Table

Thông thường bảng Calibration sẽ được nhà sản xuất cảm biến cung cấp.

---

# 6. Data Cleaning

Sau khi hiệu chuẩn, dữ liệu cần được xử lý trước khi tính đặc trưng.

Các bước gồm:
- Chuyển Timestamp sang kiểu Datetime
- Sắp xếp theo thời gian
- Loại bỏ bản ghi trùng lặp
- Xử lý Missing Values
- Đồng bộ chu kỳ lấy mẫu
- Loại bỏ giá trị ngoại lai (Outlier)
- Làm mượt tín hiệu (Moving Average hoặc Kalman Filter nếu cần)

---

# 7. Feature Engineering

Đây là bước quan trọng nhất của toàn bộ hệ thống.

Mỗi mẫu dữ liệu sẽ được chuyển thành một vector đặc trưng phục vụ cho mô hình học máy.

---

## 7.1. Delta Time

$$\Delta t = t_i - t_{i-1}$$

Đơn vị: **Second**

---

## 7.2. Fuel Difference

$$\Delta Fuel = Fuel_i - Fuel_{i-1}$$

Ý nghĩa: Lượng nhiên liệu thay đổi giữa hai lần lấy mẫu.

---

## 7.3. Fuel Rate

$$FuelRate = \frac{\Delta Fuel}{\Delta Time}$$

Đơn vị: **Liter/Second**

Đây là đặc trưng quan trọng để phát hiện Refuel và Fuel Theft.

---

## 7.4. Moving Average

Làm mượt tín hiệu. Window đề xuất: **5 mẫu**

Công thức tính (với \(W\) là kích thước cửa sổ):
$$MA_i = \frac{1}{W} \sum_{j=i-W+1}^{i} Fuel_j$$

---

## 7.5. Rolling Standard Deviation

Đo mức dao động của nhiên liệu. RollingStd lớn cho thấy hệ thống đang có biến động mạnh.

Công thức (với \(W\) là kích thước cửa sổ, và \(\mu\) là giá trị trung bình trong cửa sổ đó):
$$\sigma_{rolling} = \sqrt{\frac{1}{W} \sum_{i=1}^{W} (Fuel_i - \mu)^2}$$

---

## 7.6. Window Fuel Difference

Không nên chỉ sử dụng FuelDiff giữa hai mẫu liên tiếp.

Đề xuất thêm công thức so sánh mức nhiên liệu hiện tại với thời điểm bắt đầu cửa sổ:
$$WindowFuelDiff = Fuel_{current} - Fuel_{window\_start}$$

Ví dụ: Từ 50L → 60L → WindowFuelDiff = +10L.

Feature này phản ánh toàn bộ sự thay đổi trong cửa sổ quan sát.

---

## 7.7. Fuel Range

Cho biết mức dao động lớn nhất của nhiên liệu trong cửa sổ.

$$FuelRange = \max(Fuel_{window}) - \min(Fuel_{window})$$

---

## 7.8. Fuel Jump

Giá trị tăng hoặc giảm lớn nhất giữa các mẫu liên tiếp. Giúp phát hiện các thay đổi đột ngột.

$$FuelJump = \max_{i \in window} |Fuel_i - Fuel_{i-1}|$$

*(Trong đó \(i \in window\) đại diện cho các mẫu trong cửa sổ trượt, \(|...|\) là giá trị tuyệt đối)*.

---

## 7.9. Stop Duration

Không nên đếm số lượng mẫu mà nên tính bằng thời gian thực tính bằng giây (đơn vị: Second).

Ví dụ với Sample Interval = 5s:
```
0 → 5 → 10 → 15 → 20 (giây)
```

Công thức: \(StopDuration = t_{current} - t_{vehicle\_stop\_start}\)

---

## 7.10. Vehicle Stopped

Không nên kiểm tra `Speed == 0` mà nên dùng:

```
Speed < 2 km/h
```

để chống nhiễu cảm biến.

---

## 7.11. Acceleration

Giúp phân biệt xe đang tăng tốc hay giảm tốc.

$$Acceleration = \frac{\Delta Speed}{\Delta Time} = \frac{Speed_i - Speed_{i-1}}{\Delta t}$$

---

## 7.12. Average Speed

Trung bình tốc độ trong cửa sổ.

$$AvgSpeed = \frac{1}{W} \sum_{j=1}^{W} Speed_j$$

---

## 7.13. Time Since Last Stop

Thời gian kể từ lần dừng gần nhất (Thời gian đã trôi qua kể từ khi thoát khỏi trạng thái `VehicleStopped`).

---

## 7.14. Regression Slope (Sliding Window + Linear Regression)

Sau khi hoàn thành các bước tiền xử lý và trích xuất đặc trưng cơ bản, hệ thống không đưa ngay dữ liệu vào mô hình phân loại. Thay vào đó, một bước trung gian được thực hiện nhằm **mô tả xu hướng biến đổi của nhiên liệu theo thời gian** một cách ổn định và chống nhiễu – đó chính là tính **Regression Slope** thông qua **hồi quy tuyến tính (Linear Regression)** trên từng **cửa sổ trượt (Sliding Window)**.

### 7.14.1. Nguyên lý hoạt động

**Bước 1: Xác định kích thước cửa sổ**  
Chọn số lượng mẫu dữ liệu trong một cửa sổ, ví dụ:
```
Window Size = 10 mẫu
```
Với chu kỳ lấy mẫu `5 giây`, cửa sổ tương ứng với `50 giây`. Kích thước này đủ lớn để giảm nhiễu, nhưng vẫn đủ nhỏ để phản ứng kịp với các sự kiện đột ngột.

**Bước 2: Trượt cửa sổ**  
Khi có mẫu dữ liệu mới, cửa sổ dịch chuyển sang phải một vị trí:
```
Window 1: mẫu 1 → 10
Window 2: mẫu 2 → 11
Window 3: mẫu 3 → 12
...
```

**Bước 3: Hồi quy tuyến tính trên từng cửa sổ**  
Trong mỗi cửa sổ, ta xem thời gian `Time` là biến độc lập $X$ và mức nhiên liệu `Fuel` là biến phụ thuộc $Y$.  
Mô hình hồi quy tìm đường thẳng khớp nhất:
$$Fuel = a \times Time + b$$
với:
- $a$: **hệ số góc (Slope)** – tốc độ thay đổi nhiên liệu trung bình trong cửa sổ.
- $b$: **hệ số chặn (Intercept)** – giá trị nhiên liệu ước tính tại thời điểm $Time = 0$ của cửa sổ đó.

Từ đây, giá trị $a$ được trích xuất làm **Regression Slope** – một đặc trưng mới bổ sung vào vector đặc trưng.

### 7.14.2. Vì sao chọn hệ số góc $a$ (Slope) làm đặc trưng chính?

Linear Regression trả về đồng thời nhiều thông tin: $a$, $b$, các giá trị dự đoán, sai số (residuals) và hệ số xác định $R^2$. Tuy nhiên, **$a$ là thông tin trực tiếp trả lời câu hỏi cốt lõi của bài toán**:

> “Mức nhiên liệu đang có xu hướng **tăng**, **giảm**, hay **ổn định**?”

| Giá trị $a$ | Xu hướng nhiên liệu | Ứng dụng phân loại sự kiện |
|------------|-------------------|---------------------------|
| $a \approx 0$ | Ổn định (không đổi) | Xe đang Idle (nổ máy tại chỗ) |
| $a > 0$     | Đang tăng           | Đổ xăng (Refuel) |
| $a < 0$     | Đang giảm           | Đang chạy (Driving) hoặc bị rút trộm (Fuel Theft), cần kết hợp thêm tốc độ |

Các tham số còn lại không mang ý nghĩa phân loại trạng thái:

- **Intercept $b$** chỉ cho biết mức nhiên liệu tại thời điểm đầu cửa sổ. Một xe có $b = 50$ lít hay $20$ lít đều có thể đang chạy, đổ xăng hoặc bị trộm – $b$ không giúp phân biệt.
- **Giá trị dự đoán** và **sai số** chủ yếu dùng để đánh giá chất lượng khớp, chưa cần thiết ở bước trích xuất đặc trưng cơ bản (trừ khi mở rộng mô hình – xem mục 7.14.5).
- **$R^2$** đo mức độ tin cậy của xu hướng, hữu ích để giảm báo động giả, sẽ được đề cập sau.

### 7.14.3. So sánh Regression Slope với các phép tính đơn giản khác

**a) So với $\Delta Fuel$ (chênh lệch 2 mẫu liên tiếp)**  
$\Delta Fuel$ chỉ dùng 2 điểm, nhạy với nhiễu và không phản ánh được tốc độ.  
Ví dụ: Nhiên liệu giảm 1 lít trong 5 giây khác hoàn toàn với giảm 1 lít trong 5 phút, nhưng $\Delta Fuel$ cho cùng một giá trị.

**b) So với FuelRate ($\Delta Fuel / \Delta Time$)**  
FuelRate tính trên từng cặp điểm liền kề, dễ bị nhiễu phá hỏng. Một điểm nhiễu đơn lẻ (sai số cảm biến) có thể làm FuelRate nhảy vọt dẫn đến phân loại sai.  
Regression Slope quét toàn bộ $10$ mẫu trong cửa sổ, tìm một đường thẳng trung bình – nhờ đó **chống nhiễu rất tốt**.

Ví dụ trực quan: giả sử có chuỗi `50, 49.8, 49.7, 49.9 (nhiễu), 49.6`.  
- $\Delta Fuel$ giữa mẫu 3 và 4 là $+0.2$, có thể bị hiểu nhầm thành “đổ xăng”.  
- Regression Slope trên 5 điểm vẫn âm, phản ánh đúng xu hướng giảm.

**c) Độ trễ của cửa sổ**  
Do dùng nhiều mẫu, Regression Slope có độ trễ nhất định so với thay đổi tức thời, nhưng bù lại độ ổn định cao. Điều này hoàn toàn chấp nhận được trong bài toán giám sát với mục tiêu phát hiện sự kiện trong vòng vài chục giây.

### 7.14.4. Ý nghĩa và cách tính các tham số khác từ Linear Regression (tham khảo mở rộng)

Khi huấn luyện Linear Regression trên một cửa sổ, ngoài $a$ ta có thể thu thập thêm các giá trị sau nếu muốn tăng cường khả năng phân loại sau này:

| Tham số | Ký hiệu | Ý nghĩa | Khả năng ứng dụng |
|--------|--------|--------|-----------------|
| **Hệ số chặn** | $b$ | Mức nhiên liệu tại gốc thời gian của cửa sổ | Ít dùng vì không mô tả xu hướng. |
| **Hệ số xác định** | $R^2$ | Mức độ khớp của đường thẳng với dữ liệu (0 → 1). $R^2$ cao nghĩa là xu hướng rất rõ ràng, ít nhiễu. | Kết hợp với Slope để giảm cảnh báo giả. Ví dụ Slope dương nhưng $R^2$ thấp → có thể chỉ là nhiễu, không phải đổ xăng. |
| **Sai số toàn phương trung bình (RMSE)** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Cho biết mức độ dao động của dữ liệu quanh đường xu hướng. | Có thể thay thế hoặc bổ sung cho RollingStd. |
| **Giá trị dự đoán** | $\hat{y}$ | Mức nhiên liệu dự đoán theo đường hồi quy. | Dùng để so sánh với giá trị thực tế, phát hiện điểm bất thường (anomaly). |

Trong phiên bản đầu tiên của FuelSentinel-AI, chỉ cần **Slope ($a$)** là đủ để xây dựng baseline. Các tham số còn lại là hướng mở rộng tiềm năng nhằm cải thiện độ chính xác.

### 7.14.5. Mở rộng – Kết hợp $R^2$ vào Feature Vector

Một cải tiến nhỏ nhưng hiệu quả là đưa đồng thời **Slope ($a$)** và **$R^2$** vào vector đặc trưng.  
Khi đó, mô hình phân loại không chỉ biết xu hướng (tăng/giảm/ổn định) mà còn biết **mức độ tin cậy** của xu hướng đó.  
Ví dụ logic kết hợp:
- $a$ dương mạnh và $R^2$ cao → khả năng cao là Refuel.
- $a$ dương nhẹ và $R^2$ thấp → dữ liệu có thể đang nhiễu, tạm thời chưa kết luận.
- $a \approx 0$ và $R^2$ cao → Idle ổn định, không có biến động.
- $a$ âm mạnh và $R^2$ cao → khả năng cao là Driving hoặc Fuel Theft (phân biệt bằng Speed).

Điều này giúp giảm đáng kể tỉ lệ báo động giả do nhiễu cảm biến – một vấn đề điển hình trong hệ thống giám sát thực tế.

### 7.14.6. Tổng kết vai trò của Regression Slope trong hệ thống

- **Regression Slope** là một đặc trưng dẫn xuất từ hồi quy tuyến tính trên cửa sổ trượt, đóng vai trò then chốt trong việc mô tả **hướng và cường độ thay đổi nhiên liệu**.
- So với các đặc trưng tức thời như $\Delta Fuel$ hay FuelRate, nó có khả năng kháng nhiễu cao nhờ sử dụng toàn bộ dữ liệu trong cửa sổ.
- Lý do chỉ lấy hệ số góc $a$: vì đó là tham số duy nhất trả lời trực tiếp câu hỏi xu hướng, trong khi các tham số khác như $b$ (mức nhiên liệu gốc) không có khả năng phân biệt sự kiện, còn $R^2$ và sai số là thông tin bổ trợ có thể dùng để nâng cấp mô hình.
- Trong kiến trúc tổng thể, Regression Slope được ghép vào Feature Vector cùng các đặc trưng khác trước khi đưa qua bộ phân loại Random Forest (hoặc các mô hình cây quyết định khác).

Sau khi tính Regression Slope, danh sách đặc trưng đầy đủ của một mẫu dữ liệu sẽ được cập nhật như đã liệt kê ở Bảng đặc trưng (mục 8) và sẵn sàng cho giai đoạn huấn luyện mô hình.

---

# 8. Feature Vector

Sau khi hoàn thành toàn bộ quá trình Feature Engineering và Regression Slope, dữ liệu sẽ được mở rộng thành Feature Vector.

Ví dụ dữ liệu sau khi đã được tính toán đặc trưng:

| Timestamp | Fuel | Speed | ΔFuel | FuelRate | MovingAvg | RollingStd | FuelJump | StopDuration | Acceleration | RegressionSlope |
| --------- | ---- | ----- | ----- | -------- | --------- | ---------- | -------- | ------------ | ------------ | --------------- |
| 08:00:45  | 49.6 | 0     | 0.0   | 0.0      | 49.72     | 0.15       | 0.0      | 20           | 0            | -0.01           |
| 08:00:50  | 51.8 | 0     | 2.2   | 0.44     | 50.12     | 0.92       | 2.2      | 25           | 0            | 0.37            |
| 08:00:55  | 54.6 | 0     | 2.8   | 0.56     | 51.18     | 1.91       | 5.0      | 30           | 0            | 0.64            |
| 08:01:00  | 56.5 | 0     | 1.9   | 0.38     | 52.50     | 2.67       | 6.9      | 35           | 0            | 0.81            |
| 08:01:05  | 56.5 | 0     | 0.0   | 0.0      | 53.80     | 2.70       | 6.9      | 40           | 0            | 0.62            |
| 08:01:10  | 56.4 | 15    | -0.1  | -0.02    | 55.16     | 1.93       | 4.6      | 0            | 3.0          | 0.18            |

**Danh sách đầy đủ các đặc trưng và ý nghĩa:**

Dưới đây là bảng đặc trưng đã được chỉnh sửa, trong đó các công thức được hiển thị bằng ký hiệu toán học LaTeX thay vì mô tả dạng văn bản thông thường.

| STT | Đặc trưng (Feature) | Công thức (Formula) | Ý nghĩa (Meaning) |
|:---:|:---|:---|:---|
| 1 | **Fuel** | $F_i$ (giá trị thô sau Calibration) | Mức nhiên liệu hiện tại (Lít). |
| 2 | **Speed** | $v_i$ (giá trị thô từ GPS) | Vận tốc hiện tại của xe (km/h). |
| 3 | **FuelDiff** | $\Delta F_i = F_i - F_{i-1}$ | Chênh lệch nhiên liệu giữa 2 mẫu liên tiếp. |
| 4 | **FuelRate** | $\displaystyle r_i = \frac{\Delta F_i}{\Delta t_i} = \frac{F_i - F_{i-1}}{t_i - t_{i-1}}$ | Tốc độ thay đổi nhiên liệu (L/s). Quan trọng để phát hiện trộm cắp hoặc đổ xăng. |
| 5 | **MovingAvg** | $\displaystyle MA_i = \frac{1}{W} \sum_{j=i-W+1}^{i} F_j$ | Giá trị trung bình trượt (cửa sổ $W$ mẫu), làm mượt tín hiệu nhiễu. |
| 6 | **RollingStd** | $\displaystyle \sigma_i = \sqrt{ \frac{1}{W} \sum_{j=i-W+1}^{i} \bigl( F_j - \overline{F}_i \bigr)^2 }$ | Độ lệch chuẩn trượt, đo mức dao động của nhiên liệu trong cửa sổ (phát hiện rung động mạnh hoặc thay đổi đột ngột). |
| 7 | **WindowFuelDiff** | $\displaystyle \Delta_{\text{win}} = F_i - F_{\text{start}}$, với $F_{\text{start}}$ là mức nhiên liệu tại đầu cửa sổ trượt. | Chênh lệch nhiên liệu toàn bộ cửa sổ. Phản ánh xu hướng tổng thể (tăng/giảm ròng). |
| 8 | **FuelRange** | $\displaystyle R = \max_{j\,\in\,\text{window}} F_j - \min_{j\,\in\,\text{window}} F_j$ | Biên độ dao động nhiên liệu trong cửa sổ, phát hiện biến động cực đại. |
| 9 | **FuelJump** | $\displaystyle J = \max_{j\,\in\,\text{window}} \bigl| F_j - F_{j-1} \bigr|$ | Bước nhảy lớn nhất giữa hai mẫu liên tiếp. Rất nhạy với các sự kiện tăng/giảm đột ngột (đổ xăng, trộm cắp). |
| 10 | **Acceleration** | $\displaystyle a_i = \frac{v_i - v_{i-1}}{\Delta t_i}$ | Gia tốc của xe (km/h/s). Giúp phân biệt xe đang tăng tốc, giảm tốc hay giữ đều tốc độ. |
| 11 | **StopDuration** | $t_i - t_{\text{stop\_start}}$ | Thời gian xe dừng (tính bằng giây). Phân biệt dừng lâu (đổ xăng) với dừng ngắn (đèn đỏ). |
| 12 | **AvgSpeed** | $\displaystyle \bar{v}_i = \frac{1}{W} \sum_{j=i-W+1}^{i} v_j$ | Tốc độ trung bình trong cửa sổ. Hỗ trợ xác định trạng thái dừng/đỗ kéo dài. |
| 13 | **RegressionSlope** | $a$ trong phương trình $F = a \cdot t + b$ (hồi quy tuyến tính trên cửa sổ $W$ mẫu) | Độ dốc hồi quy tuyến tính của nhiên liệu theo thời gian. Biểu thị chính xác hướng và cường độ thay đổi nhiên liệu trong ~50 giây gần nhất. |

*Ghi chú:*  
- $i$: chỉ số của mẫu hiện tại.  
- $t_i$: thời gian tại mẫu $i$ (thường tính bằng giây).  
- $W$: kích thước cửa sổ trượt (mặc định $W = 10$ mẫu, tương đương 50 giây).  
- $\overline{F}_i$: giá trị trung bình của $F$ trong cửa sổ tại thời điểm $i$, bằng chính $MA_i$.

---

# 9. Label

Mỗi vector đặc trưng sẽ được gán nhãn. Dựa trên các đặc trưng chính như `RegressionSlope`, `FuelRate` và `Speed`:

| RegressionSlope | FuelRate | Speed | Label |
| --------------- | -------- | ----- | ----- |
| -0.02           | -0.01    | > 5   | Driving |
| 0               | 0        | 0     | Idle |
| 0.65            | 0.81     | 0     | Refuel |
| -0.84           | -0.93    | 0     | Fuel Theft |

---

# 10. Baseline Rule-based

Trước khi huấn luyện Machine Learning, cần xây dựng Rule-based để làm đường cơ sở (Baseline).

Ví dụ:
```python
if Speed > 5:
    Driving
elif VehicleStopped and WindowFuelDiff > 2:
    Refuel
elif VehicleStopped and WindowFuelDiff < -2:
    Fuel Theft
else:
    Idle
```

Rule-based giúp đánh giá xem mô hình Machine Learning có thực sự cải thiện hiệu năng hay không.

---

# 11. Machine Learning Classifier

**Đầu vào:**
```
X = [
    FuelRate,
    MovingAvg,
    RollingStd,
    FuelJump,
    FuelRange,
    Acceleration,
    StopDuration,
    RegressionSlope,
    Speed
]
```

**Đầu ra:**
```
Driving | Idle | Refuel | Fuel Theft
```

---

# 12. Các mô hình đề xuất

| Model | Khuyến nghị |
|-------|-------------|
| Decision Tree | ⭐⭐⭐⭐⭐ |
| Random Forest | ⭐⭐⭐⭐⭐ |
| Extra Trees | ⭐⭐⭐⭐ |
| Gradient Boosting | ⭐⭐⭐⭐ |
| XGBoost | ⭐⭐⭐⭐⭐ |
| LightGBM | ⭐⭐⭐⭐ |
| CatBoost | ⭐⭐⭐⭐ |

---

# 13. Không khuyến khích

- Logistic Regression
- SVM
- KNN
- Naive Bayes

Đối với giai đoạn đầu của đề tài.

LSTM, GRU hoặc Transformer chỉ nên xem xét khi có tập dữ liệu lớn và mục tiêu là dự báo hoặc phát hiện sự kiện theo chuỗi phức tạp.

---

# 14. Đánh giá mô hình

Các chỉ số nên sử dụng:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Ngoài ra, cần đánh giá:
- Thời gian suy luận (Inference Time)
- Số lượng sự kiện phát hiện đúng
- Tỷ lệ cảnh báo giả (False Alarm Rate)
- Tỷ lệ bỏ sót sự kiện (Miss Detection Rate)

---

# 15. Triển khai hệ thống

Sau khi mô hình dự đoán trạng thái, kết quả sẽ được tích hợp vào hệ thống:

- Dashboard thời gian thực
- REST API
- Event Manager
- Excel Report
- Lưu lịch sử sự kiện
- Giải thích quyết định (Explainable Timeline)

---

# 16. Pipeline hoàn chỉnh

```
                Raw Sensor Data
        (GPS, Heading, Speed, ADC)
                     │
                     ▼
         Calibration (ADC → Fuel)
                     │
                     ▼
      Data Cleaning & Noise Filtering
                     │
                     ▼
          Feature Engineering
──────────────────────────────────────────────
ΔFuel
ΔTime
FuelRate
Moving Average
Rolling Mean
Rolling Std
Fuel Jump
Stop Duration
Acceleration
Time Since Last Stop
──────────────────────────────────────────────
                     │
                     ▼
      Sliding Window (10 samples)
                     │
                     ▼
     Linear Regression trên từng cửa sổ
                     │
                     ▼
      RegressionSlope (Feature mới)
                     │
                     ▼
      Ghép vào Feature Vector cuối cùng
                     │
                     ▼
      Decision Tree / Random Forest
        / Extra Trees / XGBoost
                     │
                     ▼
Driving │ Idle │ Refuel │ Fuel Theft
                     │
                     ▼
 REST API → Dashboard → Excel Report
```

---

# 17. Khuyến nghị

Đối với phiên bản đầu tiên của FuelSentinel-AI:

- Sử dụng **Rule-based** làm baseline để xây dựng và kiểm thử pipeline.
- Lựa chọn **Random Forest** là mô hình Machine Learning chính vì cân bằng giữa độ chính xác, khả năng giải thích và chi phí tính toán.
- So sánh kết quả với **Decision Tree** (dễ diễn giải) và **XGBoost** (mô hình mạnh) để đánh giá hiệu quả của việc tăng độ phức tạp.
- Thiết kế pipeline theo hướng mô-đun (Calibration → Feature Engineering → Classification → Event Aggregation → Dashboard) để dễ dàng thay thế bộ phân loại hoặc mở rộng sang các mô hình học sâu trong tương lai mà không phải thay đổi toàn bộ kiến trúc hệ thống.