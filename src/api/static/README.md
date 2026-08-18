# FuelSentinel AI — Bản cập nhật giao diện & backend

## 1. Những gì đã đổi

### Giao diện (chỉ 2 biểu đồ: Fuel-Speed Timeline & Event Lifecycle)
- `static/css/dashboard.css`: thêm rule ép **nền trắng cố định** cho
  `.chart-container--fuel` và `.chart-container--lifecycle` (không đụng
  tới sidebar, KPI card, bảng sự kiện...). Nền trắng này giữ nguyên dù
  người dùng bật/tắt dark mode toàn trang.
- `static/js/chartBuilder.js`:
  - Đổi toàn bộ màu chữ/label/lưới trong option echarts sang tông tối
    trên nền trắng (trước đây dùng biến CSS theo theme nên khi đổi nền
    trắng chữ có thể bị "mất" trên nền tối).
  - Thêm trạng thái **`Thinking`**: khi một điểm dữ liệu chưa được xác
    nhận nhãn (đang trong giai đoạn `candidate` phía backend), vùng đó
    **để trắng, không tô màu**, và nhãn hiển thị là `Thinking…`. Ngay
    khi backend xác nhận (`confirmed`), điểm đó **đổi màu ngay** (xanh
    lá = Refuel, đỏ = Fuel Theft).
  - Trục Y nhận tham số `yAxisMax`/`yAxisMin` truyền từ `dashboard.js`
    thay vì tự tính lại mỗi lần vẽ, để hỗ trợ cơ chế "chỉ tăng, không
    giảm" bên dưới.
- `static/js/dashboard.js`:
  - **Không còn coi dashboard là "tự sống"**: chỉ khi có dữ liệu đã
    predict (`/api/points` trả về ít nhất 1 điểm) thì biểu đồ Fuel mới
    chuyển sang hiển thị dữ liệu AI (có tô màu sự kiện). Trước đó, biểu
    đồ chỉ hiển thị dữ liệu nền từ CSV (`/api/car_data`) như một
    baseline tham khảo, không có event coloring.
  - Biểu đồ Fuel: mặc định hiển thị **20 điểm gần nhất**, nạp tối đa
    **100 điểm**, vẫn dùng `dataZoom` (kéo thanh trượt) để xem các điểm
    trước/sau trong phạm vi 100 điểm đã nạp.
  - Trục Y của biểu đồ Fuel: **sàn mặc định 500**, chỉ tự tăng khi có
    điểm dữ liệu vượt quá 500 (làm tròn lên bội số 50), và **không bao
    giờ tự giảm lại** một khi đã tăng (cơ chế hysteresis, biến
    `fuelYAxisMax` ở đầu file).
  - Bảng "Event logs" và badge trạng thái hiện có thêm nhãn
    `Thinking…` khi điểm dữ liệu mới nhất chưa được xác nhận.

### Backend (`src/api/app.py`)
- **Đã xoá vòng lặp nền tự động** (`background_loop` chạy thread riêng,
  tự đọc từng dòng CSV mỗi 0.1s để "diễn" dữ liệu real-time giả lập).
  Đây là nguyên nhân khiến dashboard tự cập nhật liên tục trước đây.
- **Thêm API mới `POST /api/predict`** — đây là API DUY NHẤT khiến
  dashboard có dữ liệu mới:
  - Nhận 1 điểm hoặc 1 danh sách điểm dữ liệu (JSON) gồm `timestamp`,
    `fuel`, `speed`, `latitude`, `longitude`, `ADC` (tuỳ chọn).
  - Đẩy điểm vào buffer cửa sổ trượt (window). Khi đủ 30 mẫu mới thực
    sự chạy model CNN-GRU + cập nhật `EventTracker`.
  - Trả về ngay kết quả: nhãn dự đoán, độ tin cậy, và
    `point_status` (`thinking` | `confirmed` | `normal`) để giao diện
    biết cách tô màu.
  - Kết quả được lưu vào `processed_buffer` / `event_history` — đây là
    nguồn dữ liệu mà tất cả các API GET khác (`/api/dashboard`,
    `/api/history`, `/api/points`, `/api/tracker_status`, `/api/realtime`,
    `/api/events`, `/api/alerts`) đọc lại để trả cho giao diện.
- **Thêm API mới `GET /api/points`** — trả về các điểm ĐÃ ĐƯỢC PREDICT,
  hỗ trợ phân trang để "kéo xem trước/sau":
  - `limit`: số điểm muốn lấy (mặc định 20, tối đa 100)
  - `offset`: lùi lại bao nhiêu điểm tính từ điểm mới nhất
- **`GET /api/car_data`** giữ nguyên như cũ: đọc trực tiếp từ file
  `fusion_dataset.csv`, dùng làm dữ liệu **nền/baseline** để hiển thị
  trước khi có bất kỳ predict nào.
- Tất cả record trong `processed_buffer` giờ có thêm field
  `PointStatus` (`thinking` / `confirmed` / `normal`) để truy vết trạng
  thái xác nhận của từng điểm trong lịch sử.

## 2. Cách test nhanh cơ chế POST -> predict -> hiển thị

Vì không còn vòng lặp tự sinh dữ liệu, cần tự gửi dữ liệu vào để thấy
dashboard cập nhật. Ví dụ với `curl`:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "timestamp": "2026-07-31T10:15:00",
        "fuel": 58.2,
        "speed": 42.0,
        "latitude": 21.03,
        "longitude": 105.85,
        "ADC": 1
      }'
```

Gửi liên tục ~30 điểm (đủ `WINDOW_SIZE`) thì bắt đầu có kết quả predict
thật; trước đó API sẽ trả `{"status": "buffering", "buffered": n, "need": 30}`.

Có thể viết một script nhỏ đọc lại `fusion_dataset.csv` và POST từng
dòng lên `/api/predict` (thay cho vòng lặp nền cũ) nếu muốn mô phỏng
dữ liệu real-time từ thiết bị thật.

## 3. File không thay đổi (không có trong gói này)

- `static/js/eventSegment.js` — không cần sửa, giữ nguyên logic phân
  đoạn hiện có.
- `static/js/tooltip.js` — không có trong tài liệu gốc được cung cấp
  nên không chỉnh sửa; nếu file này có tham chiếu màu theo theme dark,
  bạn nên đổi tương tự sang tông màu tối trên nền trắng như đã làm ở
  `chartBuilder.js`.
- `templates/dashboard.html`, `templates/base.html` — không cần đổi
  cấu trúc HTML, chỉ có phần CSS/JS xử lý màu & luồng dữ liệu thay đổi.
