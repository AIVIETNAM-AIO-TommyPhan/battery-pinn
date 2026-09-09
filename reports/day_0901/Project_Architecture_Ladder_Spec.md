# Project Architecture Ladder — Spec (Track A)

## Track A vs Track B — đọc trước khi code bất cứ gì

Có 2 track tách biệt, không được trộn lẫn:

```markdown
## Track A — Project baseline track (file này)

- Input: raw tensor project [B, 6, 100, 1000] (từ feature_cache.pkl,
  BatLiNetFeatureExtractor)
- Preprocessing: DIFF_BASE=9 + clean_feature
- Model: SmallCNNScalarBranch và các biến thể kiến trúc lấy CẢM HỨNG từ
  thesis (attention pooling, per-cycle CNN, ConvLSTM1D, ConvTransformer)
  nhưng KHÔNG dùng feature representation hay target của thesis
- Mục tiêu: cải thiện RMSE/short-long bias trên đúng task hiện tại
  (dự đoán total cycle life từ cycle 0-99, so với baseline 95.42 / README 90)
- KHÔNG tuyên bố là reproduce thesis — mọi tên module "ConvLSTM",
  "ConvTransformer" trong file này là project-inspired, không phải
  implementation của thesis

## Track B — Thesis-aligned track (backlog, chưa code)

- Input: raw battery records (cần audit lại, không phải feature_cache.pkl)
- Preprocessing: channel-wise (11 timepoint × V/I/T) + cycle-wise (C/P/x3/x5)
  extraction theo đúng thesis
- Model: ConvLSTM → BiAR-SeqInSeq → BFPE-AR → ARNS → CTARNS → MvNGCN
- Mục tiêu: reproduce đúng logic progression của thesis
- **Target khác Track A**: dự đoán next-cycle capacity/SoH qua sliding
  window L=10, KHÔNG phải total life — RMSE của track này không so sánh
  được với 95.42/README=90
- Chỉ mở khi có raw-timestamp audit và feature extraction hoàn chỉnh —
  xem [Thesis_Aligned_Architecture_Ladder.md](Thesis_Aligned_Architecture_Ladder.md)
```

File này (`Project_Architecture_Ladder_Spec.md`) chỉ là **Track A**. Mọi nội dung bên dưới áp dụng cho `SmallCNNScalarBranch` và các biến thể kiến trúc trên tensor project — không phải reproduction plan của thesis.

## Phạm vi và các quyết định đã chốt

Plan này áp dụng cho `SmallCNNScalarBranch`, tức model đang là baseline xuyên suốt Tier 0–3 với top-3 scalar branch. Không áp dụng cho `SmallCNN` CNN-only.

- **Baseline tham chiếu lịch sử:** full RMSE `95.42`, nguồn `tier0_safety_loss_results.pkl[1.0]`.
- **Baseline architecture:** `SmallCNNScalarBranch`.
- **Scalar branch:** giữ trong tất cả các giai đoạn; input top-3 scalar có shape `[B, 3]`, sau MLP thành `[B, 16]` trước khi ghép với embedding tín hiệu.
- **Signal preprocessing:** mọi candidate Stage 0–3 phải dùng feature input đã qua `diff + clean_feature` với `DIFF_BASE=9`, trừ khi có một ablation preprocessing được phê duyệt rõ ràng.
- **`T_rest` không tồn tại trong processed pickle hiện tại. Chưa kết luận raw dataset không có timestamp.** Thesis off-cycle track (relaxation-effect) = `BLOCKED_UNTIL_RAW_AUDIT` — xem Track B. Track A không dùng T_rest.
- **Thực thi tuần tự:** không chạy đồng thời các job train trên GPU 4 GB.
- **Sanity gate:** Giai đoạn 0 phải reproduce baseline bằng cùng code path/protocol trước mọi can thiệp.

---

# Protocol chung (baseline-compatible)

Các giá trị sau áp dụng cho Stage 0 và là protocol mặc định cho Stage 1–3 để giữ so sánh công bằng. Bất kỳ thay đổi nào phải là ablation được ghi rõ, không được thay đổi ngầm.

```python
EPOCHS = 1000
EVAL_EVERY = 50
PATIENCE = 350  # CORRECTED 2026-09-01: historical 95.42 used patience=350, not 1e9 --
                # confirmed from variant_c_tier0_safety_loss_v1.ipynb source + Stage 0's
                # fresh seed=0 reproduction (best-val epoch 200, matched 95.42 exactly,
                # then correctly early-stopped at epoch 550). The 1e9 pin below was wrong.
OPTIMIZER = "AdamW"
LEARNING_RATE = 1e-3
# AdamW is instantiated exactly as: torch.optim.AdamW(model.parameters(), lr=1e-3)
# PyTorch default WEIGHT_DECAY = 0.01; do not pass an explicit weight_decay argument.
DIFF_BASE = 9
SCREEN_SEEDS = [0]
CONFIRM_SEEDS = [0, 1, 2, 3, 4]
SANITY_RMSE_REFERENCE = 95.42
SANITY_TOLERANCE = 0.5
```

- **Label transform:** dùng đúng transform hiện hành của `SmallCNNScalarBranch`; không được thay đổi giữa baseline và candidate. Record tên transform, các tham số fit, và inverse transform trong `config`.
- **Data split:** dùng đúng split/train/validation/test index, feature cache, top-3 scalar feature set, và preprocessing của baseline 95.42.
- **Model selection:** evaluate ở các mốc `EVAL_EVERY=50`; restore best-validation checkpoint trước khi tạo `val_pred` và `test_pred`.
- **Thiết bị:** một job train tại một thời điểm.
- **Kiểm tra shape:** mỗi class mới phải có assertion đối với channel, cycle và voltage-position dimensions trước train.

## Preprocessing bắt buộc

Bước này là baseline-defining. Không được đưa `feature_cache.pkl` raw tensor trực tiếp vào model.

```python
# feat_dev/x_signal_raw: [B, 6, 100, 1000]

DIFF_BASE = 9
reference_cycle = feat_dev[:, :, [DIFF_BASE], :]
reference_cycle: [B, 6, 1, 1000]

diffed = feat_dev - reference_cycle
diffed: [B, 6, 100, 1000]

model_input = clean_feature(diffed)
model_input: [B, 6, 100, 1000]

pred = model(model_input, scalar_dev)
```

- `clean_feature` phải được import/copy **nguyên xi** từ code gốc `run_one_scalar_safety`; không dùng placeholder hoặc implementation mới.
- Cùng `DIFF_BASE=9` và cùng `clean_feature` phải được áp dụng trước Conv2d backbone (Stage 0/1) và trước reshape sang per-cycle Conv1d encoder (Stage 2/3/backlog).

## Hàm đánh giá/gate chung

Các ngưỡng áp dụng trên **mean của 5 seed** khi xác nhận candidate. Ở screen seed 0, hàm chỉ dùng để chọn candidate đáng mở rộng, chưa đủ để công nhận thắng baseline.

```python
RMSE_TOLERANCE = 1.10
RMSE_STRONG_IMPROVEMENT = 0.95
BIAS_IMPROVEMENT_CYCLES = 15.0


def passes_primary_gate(
    rmse_new,
    short_bias_new,
    long_bias_new,
    rmse_base,
    short_bias_base,
    long_bias_base,
):
    short_gain = short_bias_base - short_bias_new
    long_gain = abs(long_bias_base) - abs(long_bias_new)
    rmse_safe = rmse_new <= rmse_base * RMSE_TOLERANCE
    rmse_strong = rmse_new <= rmse_base * RMSE_STRONG_IMPROVEMENT
    bias_gain = (
        (short_gain >= BIAS_IMPROVEMENT_CYCLES)
        or (long_gain >= BIAS_IMPROVEMENT_CYCLES)
    )
    return rmse_strong or (rmse_safe and bias_gain)
```

- `short_bias` tốt hơn khi giảm.
- `long_bias` tốt hơn khi độ lớn tuyệt đối giảm.
- Candidate chỉ được công nhận là cải thiện nếu gate pass trên mean 5-seed và `unstable=False`.
- `std` luôn bắt buộc lưu/báo cáo. Variance lớn hoặc seed failure phải tạo `unstable=True`; candidate không được chọn để deploy dù mean pass gate.

---

# Giai đoạn 0 — Sanity baseline 5-seed

## 1. Scope & tham chiếu

- **Class/model:** `SmallCNNScalarBranch`.
- **Source:** đúng class và `run_one_scalar_safety` đã tạo kết quả historical RMSE 95.42; ghi path notebook/script tuyệt đối/tương đối vào artifact.
- **Historical reference:** `tier0_safety_loss_results.pkl[1.0]`, RMSE `95.42`.
- **Mục tiêu:** reproduce baseline từ code path tương thích với historical result; xác nhận preprocessing `diff + clean_feature`, optimizer và training schedule trước mọi candidate.
- **Giữ nguyên:** model, top-3 scalar set, split, label transform, diff/clean preprocessing, optimizer instantiation, evaluation schedule, checkpoint mechanism.
- **Thay đổi:** không có thay đổi kiến trúc hay protocol.

## 2. Kiến trúc và shape chính xác

```python
# Raw cached signal
feat_raw: [B, 6, 100, 1000]

# Mandatory preprocessing
reference_cycle = feat_raw[:, :, [9], :]
reference_cycle: [B, 6, 1, 1000]
diffed = feat_raw - reference_cycle
diffed: [B, 6, 100, 1000]
x_signal = clean_feature(diffed)
x_signal: [B, 6, 100, 1000]

# CNN branch
conv1 = Conv2d(6, 16, kernel_size=3, padding=1) + ReLU
conv1_out: [B, 16, 100, 1000]

conv2 = Conv2d(16, 32, kernel_size=3, padding=1) + ReLU
conv2_out: [B, 32, 100, 1000]

pool2 = AvgPool2d(kernel_size=2)
pool2_out: [B, 32, 50, 500]

conv3 = Conv2d(32, 32, kernel_size=3, padding=1) + ReLU
conv3_out: [B, 32, 50, 500]

adaptive_pool = AdaptiveAvgPool2d((1, 1))
adaptive_pool_out: [B, 32, 1, 1]
signal_embedding = flatten(adaptive_pool_out)
signal_embedding: [B, 32]

# Scalar branch
x_scalar: [B, 3]
scalar_fc1 = Linear(3, 16) + ReLU
scalar_fc1_out: [B, 16]
scalar_fc2 = Linear(16, 16) + ReLU
scalar_embedding: [B, 16]

# Regression head
concat = cat([signal_embedding, scalar_embedding], dim=1)
concat: [B, 48]
y_hat = Linear(48, 1)(concat)
y_hat: [B, 1]
```

## 3. Ablation/biến thể

- Không có. Baseline nguyên bản.

## 4. Protocol

- `EPOCHS=1000`.
- `EVAL_EVERY=50`.
- `PATIENCE=350` (corrected 2026-09-01, see protocol chung note above).
- `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)`; không truyền `weight_decay`, nên mặc định AdamW là 0.01.
- `DIFF_BASE=9`; `diffed = clean_feature(feat_dev - feat_dev[:, :, [9], :])`.
- Seeds: `[0,1,2,3,4]`.

## 5. Tiêu chí quyết định

```python
sanity_seed0_ok = abs(rmse_seed0_fresh - 95.42) <= 0.5
sanity_multiseed_complete = all(seed in results["seeds"] for seed in [0, 1, 2, 3, 4])
stage0_ready = sanity_seed0_ok and sanity_multiseed_complete
```

Nếu `sanity_seed0_ok=False`, dừng trước Stage 1 và audit theo thứ tự: `clean_feature`, `DIFF_BASE`, scalar feature ordering, split indices, label transform, AdamW instantiation/default decay, metric code, checkpoint restoration, deterministic/CUDA settings.

## 6. Output & nơi lưu

- `ipynb/exp_v3/matr1_feature_cache/stage0_smallcnn_scalar_branch_5seed.pkl`.
- Required keys:

```python
{
    "stage": "stage0",
    "model_name": "SmallCNNScalarBranch",
    "source_model_file": "<exact path>",
    "source_runner": "run_one_scalar_safety",
    "reference_rmse": 95.42,
    "preprocessing": {
        "diff_base": 9,
        "operation": "clean_feature(feat - feat[:, :, [9], :])",
        "clean_feature_source": "<exact path/function hash if practical>",
    },
    "protocol": {
        "epochs": 1000,
        "eval_every": 50,
        "patience": 350,  # corrected 2026-09-01
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay_argument": None,
        "weight_decay_effective_default": 0.01,
        "label_transform": "<exact name/config>",
        "split_id": "<exact split/cache id>",
    },
    "seeds": {
        0: {
            "rmse": None,
            "r2": None,
            "mae": None,
            "short_bias": None,
            "long_bias": None,
            "best_epoch": None,
            "best_val_metric": None,
            "val_pred": None,
            "test_pred": None,
            "val_true": None,
            "test_true": None,
            "train_seconds": None,
            "n_parameters": None,
        }
    },
    "aggregate": {
        "rmse_mean": None,
        "rmse_std": None,
        "r2_mean": None,
        "r2_std": None,
        "mae_mean": None,
        "mae_std": None,
        "short_bias_mean": None,
        "short_bias_std": None,
        "long_bias_mean": None,
        "long_bias_std": None,
        "train_seconds_mean": None,
        "n_parameters": None,
        "unstable": None,
        "unstable_reason": None,
    },
}
```

## 7. Trạng thái dữ liệu

- Raw cached tensor: `[B,6,100,1000]`.
- Actual model input: `clean_feature(raw - raw[:, :, [9], :])`, cùng shape `[B,6,100,1000]`.
- Scalar input: top-3 features `[B,3]`.
- `T_rest` không tồn tại trong processed pickle hiện tại (chưa kết luận raw dataset không có timestamp — xem Track A vs Track B ở đầu file).
- Không thêm thesis features không tồn tại trong baseline cache/pipeline.

---

# Giai đoạn 1 — Tách trục trên SmallCNNScalarBranch

## 1. Scope & tham chiếu

- **Base class:** `SmallCNNScalarBranch`.
- **Candidate class:** `SmallCNNScalarBranchAxisAware`.
- **Baseline:** aggregate 5-seed Stage 0; 95.42 chỉ là historical reference.
- **Giữ nguyên:** preprocessing `DIFF_BASE=9 + clean_feature`, Conv2d 6→16→32, `AvgPool2d(2)`, Conv2d 32→32, scalar MLP 3→16→16, head Linear(48,1), data/split/label transform/training protocol.
- **Thay đổi duy nhất:** sau `conv3`, thay `AdaptiveAvgPool2d((1,1))` bằng pooling có cấu trúc: average trên voltage-position trước, sau đó aggregate trên cycle axis.
- **Cycle axis:** giữ `AvgPool2d(2)` để cô lập pooling-head change. Cycle length tại head là 50.

## 2. Kiến trúc chi tiết và shape

```python
feat_raw: [B, 6, 100, 1000]
x_signal = clean_feature(feat_raw - feat_raw[:, :, [9], :])
x_signal: [B, 6, 100, 1000]

conv1_out: [B, 16, 100, 1000]
conv2_out: [B, 32, 100, 1000]
pool2_out: [B, 32, 50, 500]
conv3_out: [B, 32, 50, 500]

# voltage-position aggregation
cycle_tokens_ch_first = conv3_out.mean(dim=-1)
cycle_tokens_ch_first: [B, 32, 50]
cycle_tokens = cycle_tokens_ch_first.transpose(1, 2)
cycle_tokens: [B, 50, 32]

# variants map cycle_tokens to signal_embedding
signal_embedding: [B, 32]

x_scalar: [B, 3]
scalar_embedding: [B, 16]
concat: [B, 48]
y_hat: [B, 1]
```

```python
class LearnableCyclePositionalEncoding(nn.Module):
    def __init__(self, num_cycles=50, d_model=32):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_cycles, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x):
        # x: [B, 50, 32]
        assert x.ndim == 3 and x.shape[1:] == self.pos.shape[1:]
        return x + self.pos


class CycleAttentionPooling(nn.Module):
    def __init__(self, d_model=32, hidden_dim=16):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, 50, 32]
        logits = self.score(x)                   # [B, 50, 1]
        weights = torch.softmax(logits, dim=1)  # [B, 50, 1]
        pooled = (weights * x).sum(dim=1)       # [B, 32]
        return pooled, weights.squeeze(-1)      # [B, 32], [B, 50]


class CycleConvPooling(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, 50, 32]
        z = x.transpose(1, 2)                   # [B, 32, 50]
        z = self.act(self.conv(z))              # [B, 32, 50]
        return z.mean(dim=-1)                   # [B, 32]
```

## 3. Ablation matrix

| ID | Preprocess | Backbone | Voltage aggregation | Cycle PE | Cycle aggregation | Mục tiêu |
|---|---|---|---|---|---|---|
| V0 | diff+clean, base 9 | Original | AdaptiveAvgPool2d | Không | Mean đồng thời cả 2 trục | Stage 0 reference |
| V1 | diff+clean, base 9 | Giữ đến conv3 | Mean axis 500 | Có | Attention | Full axis-aware head |
| V2 | diff+clean, base 9 | Giữ đến conv3 | Mean axis 500 | Không | Attention | Tác động PE khi attention cố định |
| V3 | diff+clean, base 9 | Giữ đến conv3 | Mean axis 500 | Có | Conv1d k=3 + mean | Attention vs local temporal conv |
| V4 | diff+clean, base 9 | Giữ đến conv3 | Mean axis 500 | Có | Mean trên 50 cycle | PE độc lập với weighted pooling |

## 4. Protocol

- Dùng protocol chung, bao gồm AdamW mặc định, full 1000 epoch, eval every 50, no early stop, diff+clean.
- Screen tuần tự: V1 seed 0 → V2 seed 0 → V3 seed 0 → V4 seed 0.
- Chỉ candidate tốt nhất pass screen gate được mở rộng 5-seed; candidate thứ hai chỉ mở rộng khi có phê duyệt rõ.

## 5. Tiêu chí quyết định

```python
screen_pass = passes_primary_gate(
    screen_metrics["rmse"],
    screen_metrics["short_bias"],
    screen_metrics["long_bias"],
    stage0["aggregate"]["rmse_mean"],
    stage0["aggregate"]["short_bias_mean"],
    stage0["aggregate"]["long_bias_mean"],
)

confirm_pass = passes_primary_gate(
    candidate5["aggregate"]["rmse_mean"],
    candidate5["aggregate"]["short_bias_mean"],
    candidate5["aggregate"]["long_bias_mean"],
    stage0["aggregate"]["rmse_mean"],
    stage0["aggregate"]["short_bias_mean"],
    stage0["aggregate"]["long_bias_mean"],
)
stage1_promote = confirm_pass and not candidate5["aggregate"]["unstable"]
```

## 6. Output & nơi lưu

- Screen: `ipynb/exp_v3/matr1_feature_cache/stage1_axis_aware_screen_seed0.pkl`.
- Confirm: `ipynb/exp_v3/matr1_feature_cache/stage1_axis_aware_confirm_5seed.pkl`.
- Thêm keys: `variant_id`, `diff_base=9`, `clean_feature_source`, `cycle_length_after_backbone=50`, `voltage_pooling`, `use_cycle_pe`, `cycle_aggregation`, `attention_weights`, `val_pred`, `test_pred`, `n_parameters`.

## 7. Trạng thái dữ liệu

- `T_rest` không tồn tại trong processed pickle hiện tại (chưa kết luận raw dataset không có timestamp — xem Track A vs Track B ở đầu file).
- Top-3 scalar branch giữ nguyên.
- Actual input luôn là diffed/cleaned tensor, không phải raw cache.

---

# Giai đoạn 2 — CNN-per-cycle + temporal aggregation

**Lưu ý:** Stage 2 (CNN-per-cycle + temporal) là project architecture track
(Track A), không phải reproduction của ConvLSTM/BiAR-SeqInSeq trong thesis.
Mọi module tên "ConvLSTM"/"attention"/"LSTM" ở đây chỉ mượn Ý TƯỞNG kiến trúc,
chạy trên tensor project `[6,100,1000]` (qua `diff+clean_feature`), không phải
channel-wise/cycle-wise representation của thesis. Reproduction thật của
thesis nằm ở Track B, xem `Thesis_Aligned_Architecture_Ladder.md`.

## 1. Scope & tham chiếu

- **Candidate class:** `CNNPerCycleScalarBranch`.
- **Thesis alignment:** SeqInSeq; encode mỗi cycle như signal 1D, rồi aggregate qua cycle sequence.
- **Baseline:** Stage 0 5-seed; báo cáo thêm so với best Stage 1 nếu Stage 1 được promote.
- **Giữ nguyên:** raw input source, `DIFF_BASE=9`, `clean_feature`, scalar branch 3→16→16, split/transform/protocol.
- **Thay đổi:** thay Conv2d backbone bằng shared Conv1d encoder chạy độc lập cho từng cycle.
- **Quyết định preprocessing:** apply full diff+clean trên tensor `[B,6,100,1000]` trước reshape per-cycle. Không diff từng cycle theo một baseline khác.
- **Quyết định về ConvLSTM (V4, bổ sung sau review):** vẫn giữ nguyên task total-life (không đổi sang rolling next-cycle forecast như thesis gốc) — V4 chỉ mượn Ý TƯỞNG kiến trúc ConvLSTM (conv+recurrent gộp trong 1 cell, giữ chiều spatial thay vì collapse trước), test công bằng bằng đúng gate/protocol như V1-V3. Bằng chứng cục bộ trước đó (`BatteryML/BatLiNet/model_convlstm.py`: ConvLSTM thua LSTM/CNN-Attn-LSTM 0/3 lần trên MATR-family data) là lý do KHÔNG ưu tiên ConvLSTM làm variant chính, nhưng không loại nó — cho nó 1 cơ hội screen như mọi variant khác.

## 2. Kiến trúc chi tiết và shape

```python
feat_raw: [B, 6, 100, 1000]
x_signal = clean_feature(feat_raw - feat_raw[:, :, [9], :])
x_signal: [B, 6, 100, 1000]

x_cycles = x_signal.permute(0, 2, 1, 3)
x_cycles: [B, 100, 6, 1000]
x_cycles = x_cycles.reshape(B * 100, 6, 1000)
x_cycles: [B*100, 6, 1000]

enc_conv1 = Conv1d(6, 16, kernel_size=7, padding=3) + ReLU
enc1_out: [B*100, 16, 1000]
enc_pool1 = AvgPool1d(2)
enc_pool1_out: [B*100, 16, 500]
enc_conv2 = Conv1d(16, 32, kernel_size=5, padding=2) + ReLU
enc2_out: [B*100, 32, 500]
enc_pool2 = AvgPool1d(2)
enc_pool2_out: [B*100, 32, 250]
enc_conv3 = Conv1d(32, 32, kernel_size=3, padding=1) + ReLU
enc3_out: [B*100, 32, 250]

# V1/V2/V3 branch: collapse spatial dim per cycle first
per_cycle_pool = AdaptiveAvgPool1d(1)
per_cycle_pool_out: [B*100, 32, 1]
cycle_embedding: [B*100, 32]
cycle_tokens: [B, 100, 32]

# V1: mean / V2: PE+attention / V3: PE+causal LSTM
signal_embedding: [B, 32]
scalar_embedding: [B, 16]
concat: [B, 48]
y_hat: [B, 1]
```

### V4 branch — `ConvLSTM1D` (project-inspired, không phải ConvLSTM reproduction của thesis), giữ chiều spatial, không collapse trước

V4 KHÔNG dùng `cycle_tokens: [B,100,32]` ở trên. Nó rẽ nhánh ngay sau `enc3_out`, giữ nguyên
chiều spatial (250) qua toàn bộ chuỗi cycle, để ConvLSTM thực sự có cấu trúc spatial+temporal
để convolve — đúng tinh thần ConvLSTM của thesis (giữ spatial per-cycle), khác với V3 (đã
collapse spatial trước khi vào LSTM thường).

```python
enc3_out: [B*100, 32, 250]
enc3_seq = enc3_out.reshape(B, 100, 32, 250)
enc3_seq: [B, 100, 32, 250]   # [batch, cycle=time, channel, spatial]

conv_lstm_hidden = ConvLSTM1D(in_channels=32, hidden_channels=32, kernel_size=3)(enc3_seq)
conv_lstm_hidden: [B, 32, 250]   # final hidden state sau 100 timestep

signal_embedding = AdaptiveAvgPool1d(1)(conv_lstm_hidden).squeeze(-1)
signal_embedding: [B, 32]
```

```python
class ConvLSTM1DCell(nn.Module):
    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv1d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=padding,
        )

    def forward(self, x_t, h_prev, c_prev):
        # x_t, h_prev, c_prev: [B, C, W]
        combined = torch.cat([x_t, h_prev], dim=1)          # [B, in+hidden, W]
        gates = self.gates(combined)                        # [B, 4*hidden, W]
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM1D(nn.Module):
    """Fused conv+recurrent cell over the cycle axis, spatial dim (voltage-position
    remnant, W=250) kept intact at every timestep -- this is what makes it a
    genuine ConvLSTM rather than CNN-then-plain-LSTM (V3)."""

    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTM1DCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x):
        # x: [B, L, C, W]  (L=100 cycles)
        assert x.ndim == 4
        B, L, C, W = x.shape
        h = x.new_zeros(B, self.hidden_channels, W)
        c = x.new_zeros(B, self.hidden_channels, W)
        for t in range(L):
            h, c = self.cell(x[:, t], h, c)
        return h  # [B, hidden_channels, W] -- last hidden state
```

Tham số: `Conv1d(64, 128, kernel_size=3)` ≈ 24.7K params — nhỏ, capacity kiểm soát, không phải
"ConvLSTM lớn trên toàn tensor" mà review trước cảnh báo tránh.

### Multi-task peak branch (thesis-inspired, optional, KHÔNG chạy mặc định)

Chỉ áp dụng cho V4 (`ConvLSTM1D`) nếu muốn thử ý tưởng multi-task của thesis
(dự đoán đồng thời RUL head chính + 1 auxiliary head), theo tinh thần chương 4
của thesis (multi-task SoH+peak). **Cảnh báo quan trọng trước khi bật:**
- "Peak" trong thesis là 1 chỉ số cụ thể của dataset gốc (NASA/CALCE) — **chưa
  có định nghĩa tương đương cho MATR** trong project này. Không được tự chế
  "peak" mà chưa xác nhận ý nghĩa/công thức tính trên dữ liệu MATR, giống hệt
  lý do T_rest bị chặn (`BLOCKED_UNTIL_RAW_AUDIT`).
- "SoH head" ở đây trùng ý tưởng với Tier 1 (SOH auxiliary head) của track
  loss-shaping/PINN riêng biệt (đã tạm dừng, xem `day_0827`) — nếu bật, nên
  đối chiếu lại kết quả Tier 1 (không cải thiện rõ so với baseline) trước khi
  kỳ vọng nhiều.

```markdown
- SoH head: [B, hidden] → [B, 1]   (hidden = signal_embedding, [B,32] từ V4)
- Peak head: [B, hidden] → [B, 1]  (CHƯA ĐỊNH NGHĨA "peak" cho MATR)
- Loss: L = alpha * L_SoH + (1 - alpha) * L_Peak

Biến thể (chỉ mở sau khi V4 single-task đã screen xong VÀ "peak" đã được
định nghĩa/audit cho MATR):
- CL-1: ConvLSTM1D single-task (= V4 ở trên)
- CL-2: ConvLSTM1D multi-task SoH + peak
```

Không thuộc screen bắt buộc của Stage 2 — chỉ mở nếu V4 single-task cho tín
hiệu và "peak" đã được audit định nghĩa xong.

## 3. Ablation matrix

| ID | Preprocess | Per-cycle encoder | PE | Aggregation |
|---|---|---|---|---|
| V1 | diff+clean, base 9 | Conv1d 6→16→32→32 | Không | Mean 100 cycles |
| V2 | diff+clean, base 9 | Conv1d 6→16→32→32 | Learnable PE | Attention pooling |
| V3 | diff+clean, base 9 | Conv1d 6→16→32→32 | Learnable PE | 1-layer causal LSTM, hidden=32 |
| V4 | diff+clean, base 9 | Conv1d 6→16→32→32 (không collapse spatial) | Không | `ConvLSTM1D` (project-inspired), hidden_channels=32, giữ spatial W=250 |

`LSTM + attention` không chạy vòng đầu; chỉ được mở sau khi V3 pass 5-seed và có phê duyệt.
V4 (ConvLSTM) được screen bình đẳng cùng V1-V3, seed=0, cùng gate — không ưu tiên, không loại
trừ trước. Bằng chứng cục bộ (`BatteryML/BatLiNet/model_convlstm.py`: ConvLSTM thua LSTM/CNN-
Attn-LSTM 0/3 lần trên MATR-family data) là lý do KHÔNG mặc định kỳ vọng V4 thắng, không phải
lý do loại bỏ nó khỏi ablation.

## 4. Protocol

- Protocol chung, không đổi preprocessing/optimizer/schedule.
- Screen: V1 seed 0 → V2 seed 0 → V3 seed 0 → V4 seed 0.
- Best screen candidate pass gate → 5 seed confirmation.

## 5. Tiêu chí quyết định

Dùng `passes_primary_gate(...)` so với Stage 0 aggregate; promote chỉ khi mean 5 seed pass và không unstable.

## 6. Output & nơi lưu

- Screen: `ipynb/exp_v3/matr1_feature_cache/stage2_per_cycle_temporal_screen_seed0.pkl`.
- Confirm: `ipynb/exp_v3/matr1_feature_cache/stage2_per_cycle_temporal_confirm_5seed.pkl`.
- Required additional keys: `variant_id`, `preprocessing`, `diff_base`, `per_cycle_shape=[6,1000]`, `n_cycles=100`, `embedding_dim=32`, `temporal_module`, `use_cycle_pe`, `attention_weights`, `val_pred`, `test_pred`, `n_parameters`.
- V4 (ConvLSTM1D) thêm: `spatial_width_kept=250` (xác nhận V4 không collapse spatial như V1-V3), `conv_lstm_hidden_channels=32`, `conv_lstm_kernel_size=3`.

## 7. Trạng thái dữ liệu

- `T_rest` không tồn tại trong processed pickle hiện tại (chưa kết luận raw dataset không có timestamp — xem Track A vs Track B ở đầu file).
- top-3 scalar branch giữ nguyên.
- Diff/clean diễn ra trước per-cycle reshape.

---

# Giai đoạn 3 — ConvTransformer-inspired, giới hạn capacity

## 1. Scope & tham chiếu

- **Candidate class:** `PerCycleConvTransformerScalarBranch`.
- **Thesis alignment:** lấy cảm hứng CTARNS: temporal convolution + self-attention, nhưng không sao chép nguyên xi CTARNS do current feature tensor/target/protocol khác thesis.
- **Baseline:** Stage 0 5-seed; báo cáo thêm vs best promoted Stage 1/2.
- **Giữ nguyên:** diff+clean base 9, Conv1d per-cycle compression, scalar MLP, split/transform/training protocol.
- **Thay đổi:** temporal aggregation Stage 2 được thay bằng lightweight ConvTransformer block trên token `[B,100,32]`.
- **Không sử dụng:** `T_rest` hoặc off-cycle feature không có trong cache (không tồn tại trong processed pickle hiện tại; chưa kết luận raw dataset không có timestamp — xem Track A vs Track B ở đầu file).

## 2. Kiến trúc chi tiết và shape

```python
feat_raw: [B, 6, 100, 1000]
x_signal = clean_feature(feat_raw - feat_raw[:, :, [9], :])
x_signal: [B, 6, 100, 1000]

# Shared Stage-2 Conv1d encoder
cycle_tokens: [B, 100, 32]
cycle_tokens_pe: [B, 100, 32]

# Causal temporal Conv1d
z: [B, 32, 100]
z_padded: [B, 32, 104]
tconv_out: [B, 32, 100]
tconv_tokens: [B, 100, 32]
z1 = LayerNorm(32)(cycle_tokens_pe + tconv_tokens)
z1: [B, 100, 32]

# Self-attention
attn_out: [B, 100, 32]
attn_map: [B, 100, 100]
z2 = LayerNorm(32)(z1 + attn_out)
z2: [B, 100, 32]

# Pointwise FFN
ffn_out: [B, 100, 32]
z3 = LayerNorm(32)(z2 + ffn_out)
z3: [B, 100, 32]

signal_embedding = z3.mean(dim=1)
signal_embedding: [B, 32]
scalar_embedding: [B, 16]
concat: [B, 48]
y_hat: [B, 1]
```

## 3. Ablation matrix

| ID | Preprocess | Temporal Conv | Self-attention | PE | Token pooling |
|---|---|---|---|---|---|
| V1 | diff+clean, base 9 | Causal Conv1d k=5 | Không | Có | Mean |
| V2 | diff+clean, base 9 | Không | 4-head attention | Có | Mean |
| V3 | diff+clean, base 9 | Causal Conv1d k=5 | 4-head attention | Có | Mean |

Fixed capacity: `d_model=32`, `num_heads=4`, `ffn_dim=64`, `n_blocks=1`, temporal `kernel_size=5`, `dilation=1`.

## 4. Protocol

- Protocol chung, including full 1000 epochs and AdamW default decay.
- Screen sequential V1 seed 0 → V2 seed 0 → V3 seed 0.
- Best pass candidate → 5 seed confirmation.
- Lưu parameter count, `train_seconds`, peak GPU memory nếu runner có hỗ trợ.

## 5. Tiêu chí quyết định

Dùng `passes_primary_gate(...)` vs Stage 0 5-seed aggregate. Train time chỉ là secondary report metric, không phải accuracy gate.

## 6. Output & nơi lưu

- Screen: `ipynb/exp_v3/matr1_feature_cache/stage3_convtransformer_screen_seed0.pkl`.
- Confirm: `ipynb/exp_v3/matr1_feature_cache/stage3_convtransformer_confirm_5seed.pkl`.
- Additional keys: `variant_id`, `preprocessing`, `diff_base`, `d_model=32`, `num_heads=4`, `n_blocks=1`, `temporal_conv`, `attention_maps`, `n_parameters`, `val_pred`, `test_pred`.

## 7. Trạng thái dữ liệu

- `T_rest` không tồn tại trong processed pickle hiện tại (chưa kết luận raw dataset không có timestamp — xem Track A vs Track B ở đầu file).
- top-3 scalar branch giữ nguyên.
- diff+clean bắt buộc trước per-cycle encoder.

---

# Backlog tùy chọn — GCN/MvNGCN-inspired

## Điều kiện mở backlog

```python
open_gcn_backlog = (
    (stage2_promote or stage3_promote)
    and graph_hypothesis_documented
    and user_approval_received
)
```

GCN không phải bước bắt buộc. Chỉ mở khi evidence từ Stage 2/3 cho thấy non-local relation giữa cycle tokens thực sự hữu ích, không thể giải thích tương đương bởi local temporal conv/attention.

## 1. Scope & kiến trúc

- **Candidate class:** `CycleGraphScalarBranch`.
- **Giữ nguyên:** diff+clean base 9, Stage-2 Conv1d per-cycle encoder, scalar branch, head, split/protocol.
- **Thay đổi:** thay temporal aggregator bằng learned adjacency graph convolution trên 100 cycle nodes.

```python
feat_raw: [B, 6, 100, 1000]
x_signal = clean_feature(feat_raw - feat_raw[:, :, [9], :])
cycle_tokens: [B, 100, 32]
Q = Linear(32,16)(cycle_tokens): [B,100,16]
K = Linear(32,16)(cycle_tokens): [B,100,16]
A = softmax(Q @ K.transpose(-1,-2) / sqrt(16), dim=-1): [B,100,100]
H = ReLU(A @ Linear(32,32)(cycle_tokens)): [B,100,32]
signal_embedding = H.mean(dim=1): [B,32]
scalar_embedding: [B,16]
concat: [B,48]
y_hat: [B,1]
```

## 2. Ablation

| ID | Adjacency | Graph layers | Readout |
|---|---|---|---|
| V1 | Fixed local chain | 1 | Mean |
| V2 | Learned QK adjacency | 1 | Mean |
| V3 | Learned QK + residual | 2 maximum | Mean; only after V2 confirm pass |

## 3. Protocol, gate, output

- Dùng protocol chung và `passes_primary_gate(...)` vs Stage 0.
- Screen: V1 seed 0 → V2 seed 0; V3 chỉ sau V2 5-seed pass.
- Confirm: 5 seed, non-unstable bắt buộc.
- Screen: `ipynb/exp_v3/matr1_feature_cache/backlog_cycle_gcn_screen_seed0.pkl`.
- Confirm: `ipynb/exp_v3/matr1_feature_cache/backlog_cycle_gcn_confirm_5seed.pkl`.
- Keys: `preprocessing`, `diff_base`, `adjacency_type`, `adjacency_matrices`, `graph_layers`, `n_parameters`, `val_pred`, `test_pred`.

---

# Giai đoạn 4 — Tổng hợp và quyết định

## Input artifacts

```text
stage0_smallcnn_scalar_branch_5seed.pkl
stage1_axis_aware_screen_seed0.pkl
stage1_axis_aware_confirm_5seed.pkl (nếu có)
stage2_per_cycle_temporal_screen_seed0.pkl
stage2_per_cycle_temporal_confirm_5seed.pkl (nếu có)
stage3_convtransformer_screen_seed0.pkl
stage3_convtransformer_confirm_5seed.pkl (nếu có)
backlog_cycle_gcn_* (nếu backlog được mở)
```

## Required summary fields

| Field | Ý nghĩa |
|---|---|
| model/stage/variant | Nhận diện experiment |
| seed set | Screen hoặc 5-seed confirm |
| RMSE mean ± std | Primary accuracy |
| MAE mean ± std | Auxiliary accuracy |
| R2 mean ± std | Goodness-of-fit |
| short_bias mean ± std | Bias short horizon |
| long_bias mean ± std | Bias long horizon |
| best epoch mean ± std | Convergence/checkpoint behavior |
| parameter count | Capacity control |
| training seconds mean ± std | Cost report |
| preprocessing | `DIFF_BASE` + `clean_feature` provenance |
| pass gate | Boolean |
| unstable/reason | Stability decision |

## Selection logic

```python
eligible = (
    result["aggregate"]["pass_primary_gate"]
    and not result["aggregate"]["unstable"]
)

# Sort eligible candidates by:
# 1. lowest rmse_mean
# 2. largest short-bias gain
# 3. largest long-bias gain
# 4. lower parameter count
# 5. lower train_seconds_mean
```

## Outputs

- `ipynb/exp_v3/matr1_feature_cache/stage4_longterm_plan_summary.pkl`.
- `ipynb/exp_v3/matr1_feature_cache/stage4_longterm_plan_summary.md`.

---

# Quyết định về model thư viện có sẵn

`CNNRULPredictor`, `LSTMRULPredictor`, `TransformerRULPredictor` không nằm trong execution path mặc định. Đây là quyết định chủ đích: dùng trực tiếp `6×1000=6000` features/cycle gây capacity lớn, không phù hợp mục tiêu stability với train set nhỏ và lịch sử seed instability. Stage 2/3 dùng Conv1d per-cycle encoder nén thành token 32 chiều trước temporal model.

---

# Thứ tự thực thi và compute accounting

```text
Stage 0: baseline 5 seed + sanity gate
    ↓ only if seed-0 sanity passes
Stage 1: V1/V2/V3/V4 seed 0, sequential
    ↓ best screen candidate passes
Stage 1: confirmation 5 seed
    ↓ promote or continue
Stage 2: V1/V2/V3 seed 0, sequential
    ↓ best screen candidate passes
Stage 2: confirmation 5 seed
    ↓ if needed
Stage 3: V1/V2/V3 seed 0, sequential
    ↓ best screen candidate passes
Stage 3: confirmation 5 seed
    ↓ optional GCN only with graph hypothesis + approval
Stage 4: summary and reproducible report
```

Track compute bằng `số run × train_seconds thực tế`, không ước lượng bằng ngày.

- Historical single-run estimate khoảng 1300–1500 giây (22–25 phút), cần cập nhật bằng `train_seconds` Stage 0.
- Stage 0: 5 run.
- Stage 1 screen: 4 run; confirm: tối đa 5 run cho candidate đã chọn.
- Stage 2 screen: 4 run (V1/V2/V3/V4, V4 = ConvLSTM1D bổ sung sau review thesis); confirm: tối đa 5 run cho candidate đã chọn.
- Stage 3 screen: 3 run; confirm: tối đa 5 run cho candidate đã chọn.
- Backlog GCN chỉ chạy nếu thỏa điều kiện mở backlog.
