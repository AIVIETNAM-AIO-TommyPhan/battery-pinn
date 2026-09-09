# Thesis-Aligned Architecture Ladder — Track B (Backlog, chưa code)

## Cảnh báo quan trọng nhất — đọc trước khi mở track này

**Track này dự đoán một target KHÁC với Track A (`Project_Architecture_Ladder_Spec.md`).**

- **Track A (project)**: cho model nhìn cycle 0-99 → đoán **tổng cycle life**
  (1 số duy nhất cho cả đời pin). So sánh với baseline 95.42 và README=90.
- **Track B (thesis, file này)**: cho model nhìn `L=10` cycle gần nhất → đoán
  **capacity/SoH của cycle kế tiếp** (`C[k+11]`), theo kiểu rolling forecast.

Hai task này **không thể so sánh RMSE trực tiếp** — khác đơn vị, khác scale,
khác ý nghĩa. Track B không "thay thế" hay "cải thiện" con số 95.42 — nó là
một câu hỏi nghiên cứu riêng (rolling SoH forecasting), chỉ nên mở nếu mục
tiêu thực sự là trả lời câu hỏi đó, không phải để tìm backbone tốt hơn cho
bài toán total-life hiện tại (đó là việc của Track A).

**Điều kiện mở track này:** người dùng xác nhận rõ ràng muốn theo đuổi câu hỏi
rolling-forecast như một nhánh nghiên cứu độc lập, VÀ đã audit xong dữ liệu
raw (B0) xác nhận đủ field cần thiết. Không tự động mở chỉ vì thesis có sẵn
kiến trúc này.

## Vì sao ConvLSTM không tự động là "phase 1 bắt buộc"

Bằng chứng cục bộ trong chính project: `BatteryML/BatLiNet/model_convlstm.py`
ghi rõ ConvLSTM đã thua LSTM và CNN-Attn-LSTM 0/3 lần trong ablation trên
MATR-family data. Số cải thiện của thesis (báo cáo ~0.045→0.033, thang đo
SoH-fraction) đến từ dataset/task khác (NASA/CALCE, rolling SoH forecast),
không dự đoán được việc ConvLSTM sẽ tốt trên MATR/task total-life. Track A đã
cho ConvLSTM 1 cơ hội công bằng (Stage 2's V4 `ConvLSTM1D`) mà không cần mở
toàn bộ track này.

## Outline các bước (B0-B9) — chỉ ghi lại, chưa spec chi tiết như Track A

```text
B0: Raw-data audit
    - Xác nhận raw MATR records (trước khi qua BatteryML processing) có
      absolute timestamp hay không -- đây là điều kiện tiên quyết để có
      T_rest thật, khác với "processed pickle không có" đã xác nhận ở Track A.
    - Xác nhận field nào tương đương "peak" (P), x3, x5 của thesis tồn tại
      hoặc tính được từ raw MATR data.
    - Nếu B0 không xác nhận được timestamp/field cần thiết, dừng track này,
      không tự chế feature giả.

B1: Channel-wise/cycle-wise feature extraction
    - Định nghĩa chính xác "11 timepoint" lấy ở đâu trong mỗi cycle (theo
      charging phase, theo % SOC, hay cố định theo index) -- đây là quyết
      định thiết kế cần làm rõ, thesis không tự động áp dụng được cho MATR.
    - Leakage audit bắt buộc: scaler chỉ fit trên train, C_k là capacity quá
      khứ không phải target, tách theo battery không theo cycle.

B2: MLP + MC-LSTM baseline (cycle-wise/channel-wise)
    - Mục tiêu: xác nhận feature pipeline B1 chạy đúng, KHÔNG cần thắng
      SmallCNN/Track A -- 2 track không so sánh trực tiếp được (xem cảnh báo
      ở trên).

B3: ConvLSTM single-task (đúng nghĩa thesis: channel-wise [B,10,11,3])
B4: ConvLSTM multi-task (SoH + peak head)
B5: BiAR-SeqInSeq (nested BiLSTM local + cycle-wise + AR branch)
B6: BFPE-AR (fixed-point/trainable positional encoding + self-attention + AR)
B7: ARNS (thêm T_rest -- CHỈ nếu B0 xác nhận raw timestamp tồn tại)
B8: CTARNS (ConvTransformer thay nested BiLSTM)
B9: MvNGCN backlog (channel-view + cycle-view graph, chỉ mở nếu có graph
    hypothesis cụ thể + phê duyệt)
```

## Trạng thái

Chưa bắt đầu B0. Không code gì trong track này cho tới khi có xác nhận rõ
ràng muốn mở (xem "Điều kiện mở track này" ở trên).
