# WORKFLOW DỰ ÁN: RAMAN PHYSICS-INFORMED AI

## Tổng quan kiến trúc hội tụ (Convergence Architecture)

Hệ thống được thiết kế theo mô hình **cây hội tụ (Convergence Tree)** — 4 nhánh phát triển song song trong giai đoạn đầu, nối tiếp khi cần đầu ra của nhánh trước, và hội tụ qua một nút tích hợp (Integration Node) để tạo ra sản phẩm cuối cùng: **Target Model + Project Directory**.

### Triết lý thiết kế

Mọi quyết định kiến trúc đều tuân theo nguyên tắc: *"Model không được phép học bất kỳ điều gì vi phạm định luật vật lý."* Không gian tham số của mạng nơ-ron bị ràng buộc (constrained) bởi các phương trình cơ học lượng tử và quang phổ học, đảm bảo mọi kết quả đầu ra đều có thể diễn dịch được về mặt hóa lý.

### Sơ đồ phụ thuộc giữa các nhánh

```
Nhánh 1 (Groundwork) ──────────────────────────────────┐
   │ Đầu ra: Thư viện phương trình, loss constraints   │
   │                                                     │
Nhánh 2 (Gold Standard) ────────────────────────────────┤──► Integration Node ──► Validation ──► TARGET MODEL
   │ Đầu ra: Dataset, metrics                           │
   │                                                     │
Nhánh 3 (Core Engine) ◄── [cần N1 + N2] ───────────────┤
   │ Đầu ra: Model pipeline                             │
   │                                                     │
Nhánh 4 (Interface) ◄── [cần N3] ──────────────────────┘
   │ Đầu ra: XAI + Automation
```

**Quy tắc phụ thuộc:**
- Nhánh 1 và 2: song song hoàn toàn, không phụ thuộc lẫn nhau.
- Nhánh 3: yêu cầu đầu ra của cả Nhánh 1 (constraints, loss functions) và Nhánh 2 (data, metrics) trước khi bắt đầu huấn luyện.
- Nhánh 4: yêu cầu đầu ra của Nhánh 3 (model đã huấn luyện) để xây dựng XAI và automation pipeline.

---

## NHÁNH 1: THE GROUNDWORK — Phân tích nền tảng & Chứng minh lỗ hổng

**Mục tiêu:** Xây dựng bộ khung lý thuyết, xác định các rào cản vật lý (physics constraints) mà model không được phép vi phạm, và chứng minh tại sao cần một hệ thống Physics-Informed.

**Timeline dự kiến:** Tuần 1–4

### Task 1.1: SOTA Review & Baseline Evaluation

**Mô tả chi tiết:**
Thực hiện tổng quan hệ thống (systematic review) các phương pháp chemometrics truyền thống và deep learning hiện có cho phân tích quang phổ Raman. Mục tiêu không chỉ là liệt kê mà phải thiết lập một baseline hiệu năng (performance baseline) để so sánh với model cuối cùng.

**Các bước thực hiện:**

1. **Khảo sát Chemometrics truyền thống:**
   - Thu thập và thực hiện lại (reproduce) các pipeline tiêu chuẩn: PCA → SVM, PLS-DA, LDA trên ít nhất 2 bộ dữ liệu Raman benchmark (ví dụ: bộ dữ liệu phân loại vi khuẩn, bộ dữ liệu phân biệt mô ung thư).
   - Ghi nhận chi tiết các bước tiền xử lý truyền thống: Savitzky-Golay smoothing, airPLS baseline correction, cosmic ray removal, vector normalization.
   - Đo hiệu năng: Accuracy, F1-Score (macro/micro), Confusion Matrix, thời gian xử lý trên mỗi phổ.
   - Ghi lại tất cả hyperparameters (số thành phần PCA, kernel SVM, số latent variables PLS-DA).

2. **Khảo sát Deep Learning hiện tại:**
   - Thực hiện lại ít nhất 3 kiến trúc đại diện: 1D-CNN, ResNet-1D, Vision Transformer (ViT) biến thể 1D trên cùng bộ dữ liệu.
   - So sánh hiệu năng với baseline truyền thống ở trên.
   - Phân tích Grad-CAM / Saliency Maps để xem model đang "nhìn" vào vùng nào của phổ.

3. **Viết báo cáo SOTA:**
   - Bảng so sánh hiệu năng có hệ thống (systematic comparison table).
   - Xác định rõ các giới hạn (limitations) của từng phương pháp.
   - Xác định gaps mà dự án này nhắm tới.

**Đầu ra:** File `docs/sota_review.md`, bảng benchmark `benchmark/baseline_results.csv`, code reproduce trong `benchmark/baselines/`.

---

### Task 1.2: Chứng minh tính "Black Box" của Deep Learning hiện tại

**Mô tả chi tiết:**
Thiết kế thí nghiệm có đối chứng (controlled experiment) để chứng minh rằng các mô hình deep learning hiện tại không "hiểu" bản chất cấu trúc phân tử mà chỉ học các tương quan thống kê bề mặt.

**Các bước thực hiện:**

1. **Adversarial Attack trên phổ Raman:**
   - Lấy model 1D-CNN đã huấn luyện ở Task 1.1.
   - Thêm nhiễu Gaussian cực nhỏ (σ < 1% cường độ trung bình) vào vùng phổ KHÔNG chứa thông tin hóa học (vùng nền, ngoài dải 400–1800 cm⁻¹) → Quan sát xem model có thay đổi dự đoán không.
   - Thêm nhiễu có mục đích vào đúng vị trí các đỉnh phổ quan trọng (ví dụ: dịch đỉnh Amide I từ 1650 cm⁻¹ sang 1660 cm⁻¹) → Model phải thay đổi dự đoán. Nếu không, chứng tỏ model không hiểu ý nghĩa của peak.

2. **Thí nghiệm "Physics Violation":**
   - Tạo phổ giả có hình dạng hoàn toàn không tuân theo profil Lorentzian/Voigt (ví dụ: đỉnh hình tam giác, hình vuông) → Đưa vào model và ghi nhận: model có vẫn đưa ra kết quả tự tin (high confidence) không?
   - Nếu model vẫn phân loại chính xác hoặc tự tin → Đây là bằng chứng rõ ràng rằng model không encode kiến thức vật lý.

3. **Phân tích Feature Attribution:**
   - Sử dụng SHAP (SHapley Additive exPlanations) để xác định top-N features (wavenumber regions) mà model dựa vào.
   - So sánh với bản đồ kiến thức hóa lý (Task 1.3): model có đang nhìn vào đúng vị trí liên kết hóa học hay đang dựa vào artifacts?

**Đầu ra:** File `docs/groundwork.md` phần "Black Box Demonstration", notebook `notebooks/adversarial_analysis.ipynb`.

---

### Task 1.3: Bản đồ phương trình Toán-Lý-Hóa-Sinh của Raman

**Mô tả chi tiết:**
Xây dựng một thư viện toán học đầy đủ (mathematical library) chứa tất cả các phương trình lý thuyết liên quan đến quang phổ Raman. Thư viện này sẽ được sử dụng trực tiếp trong loss function và constraints của model.

**Các bước thực hiện:**

1. **Phương trình tán xạ cơ bản:**
   - Cường độ tán xạ Raman: $I_{Raman} \propto \nu_0^4 \cdot |\alpha'|^2 \cdot (n_i + 1)$
     trong đó $\nu_0$ là tần số ánh sáng kích thích, $\alpha'$ là đạo hàm polarizability, $n_i$ là số hạt phonon (phụ thuộc nhiệt độ qua phân bố Bose-Einstein).
   - Tỷ số Stokes/Anti-Stokes: $\frac{I_{AS}}{I_S} = \left(\frac{\nu_0 + \nu_m}{\nu_0 - \nu_m}\right)^4 \exp\left(-\frac{h\nu_m}{k_BT}\right)$
   - Mã hóa thành hàm Python có thể tính gradient (differentiable) bằng PyTorch/JAX.

2. **Hàm hình dạng đỉnh phổ (Peak Profile Functions):**
   - Lorentzian: $L(\nu) = \frac{A \cdot \Gamma^2}{(\nu - \nu_0)^2 + \Gamma^2}$
   - Gaussian: $G(\nu) = A \cdot \exp\left(-\frac{(\nu - \nu_0)^2}{2\sigma^2}\right)$
   - Voigt (tích chập Lorentzian × Gaussian): $V(\nu) = \int_{-\infty}^{\infty} G(\nu') \cdot L(\nu - \nu') d\nu'$
   - Pseudo-Voigt (xấp xỉ nhanh): $pV(\nu) = \eta \cdot L(\nu) + (1 - \eta) \cdot G(\nu)$ với $\eta \in [0,1]$
   - Mỗi hàm phải differentiable qua PyTorch autograd.

3. **Bản đồ Symbolic Knowledge (Wavenumber → Chemical Bond):**
   - Xây dựng dictionary chuẩn ánh xạ vùng số sóng → kiểu dao động → liên kết hóa học:
     - 500–700 cm⁻¹: C-S stretch, S-S stretch
     - 700–900 cm⁻¹: C-H out-of-plane bending (aromatic)
     - 900–1100 cm⁻¹: C-O-C stretch, PO₄³⁻ symmetric stretch
     - 1000–1010 cm⁻¹: Phenylalanine ring breathing (marker sinh học)
     - 1200–1300 cm⁻¹: Amide III (protein secondary structure)
     - 1440–1460 cm⁻¹: CH₂ scissoring (lipids)
     - 1550–1570 cm⁻¹: Amide II (chủ yếu thấy ở IR, ít ở Raman)
     - 1650–1680 cm⁻¹: Amide I (C=O stretch, marker chính cho cấu trúc bậc 2 protein)
     - 2800–3000 cm⁻¹: C-H stretch (lipids, hydrocarbons)
     - 3200–3500 cm⁻¹: O-H stretch, N-H stretch
   - Mã hóa thành file JSON/YAML cấu trúc để model có thể truy vấn (queryable).

4. **Các ràng buộc vật lý (Physics Constraints) cho Loss Function:**
   - **Non-negativity constraint:** Cường độ phổ Raman phải ≥ 0 tại mọi điểm.
   - **Peak profile constraint:** Mọi đỉnh phổ phát hiện được phải fit vào profil Lorentzian hoặc Voigt — không có hình dạng vật lý vô nghĩa.
   - **Conservation constraint:** Tổng diện tích phổ phải bảo toàn sau tiền xử lý (không được "mất" hoặc "tạo" tín hiệu).
   - **Temperature consistency (nếu có Stokes + Anti-Stokes):** Tỷ số cường độ phải nhất quán với nhiệt độ mẫu.
   - **Spectral smoothness:** Đạo hàm bậc hai của phổ không được có các bước nhảy đột ngột phi vật lý.
   - Mã hóa mỗi constraint thành một hàm penalty differentiable.

5. **Mở rộng cho IR (hướng tương lai):**
   - Ghi chú các quy tắc chọn lọc (selection rules): Raman-active vs IR-active modes (nguyên tắc loại trừ tương hỗ trong phân tử có tâm đối xứng).
   - Chuẩn bị interface cho việc thêm phương trình Beer-Lambert (cho IR quantitative analysis).

**Đầu ra:**
- `engine/physics_laws.py`: Các hàm toán học differentiable (Lorentzian, Voigt, Rayleigh/Raman scattering).
- `engine/symbolic_math.py`: Dictionary ánh xạ wavenumber → chemical bonds.
- `engine/constraints.py`: Các hàm penalty/constraint cho loss function.
- `engine/spectral_database.yaml`: Cơ sở dữ liệu symbolic knowledge ở dạng có cấu trúc.
- `docs/groundwork.md` phần "Mathematical Framework".

---

## NHÁNH 2: THE GOLD STANDARD — Xây dựng Benchmark chuyên biệt

**Mục tiêu:** Tạo ra bộ dữ liệu chất lượng cao và hệ thống đánh giá chuyên biệt cho quang phổ Raman, vượt xa các metrics đánh giá ML thông thường.

**Timeline dự kiến:** Tuần 1–6 (song song với Nhánh 1 từ Tuần 1–4, tiếp tục đến Tuần 6)

### Task 2.1: Thu thập và Chuẩn hóa dữ liệu thực (Real Data Pipeline)

**Mô tả chi tiết:**
Xây dựng pipeline thu thập, tiền xử lý, và kiểm tra chất lượng dữ liệu phổ Raman thực nghiệm.

**Các bước thực hiện:**

1. **Xác định nguồn dữ liệu:**
   - Dữ liệu từ phòng thí nghiệm: phổ Raman của tế bào, mô sinh học, vật liệu polymer, chất bán dẫn, v.v.
   - Dữ liệu công khai (public datasets):
     - RRUFF database (khoáng vật)
     - Bio-Raman datasets từ các nhóm nghiên cứu (tế bào ung thư, vi khuẩn)
     - SERS datasets (Surface-Enhanced Raman Spectroscopy)
   - Ghi nhận đầy đủ metadata: bước sóng laser kích thích (532 nm, 633 nm, 785 nm), thời gian tích phân, công suất laser, loại detector.

2. **Quy trình kiểm tra chất lượng (Quality Control):**
   - Loại bỏ phổ bị cosmic ray chưa xử lý.
   - Kiểm tra tỷ lệ SNR (Signal-to-Noise Ratio): loại bỏ phổ có SNR < ngưỡng tối thiểu (tuỳ ứng dụng, ví dụ SNR < 5).
   - Kiểm tra baseline: phổ có baseline quá cao (fluorescence) cần được đánh dấu.
   - Kiểm tra phổ bất thường (outlier) bằng PCA — phổ nằm ngoài 3σ cần kiểm tra thủ công.

3. **Chuẩn hóa format:**
   - Thống nhất tất cả phổ về cùng trục X: 100–4000 cm⁻¹, bước 1 cm⁻¹ (hoặc resolution gốc nếu Neural Operator xử lý được arbitrary resolution).
   - Format lưu trữ: `.npy` (NumPy array) cho phổ, `.json` cho metadata.
   - Cấu trúc thư mục: `data/raw/{dataset_name}/{sample_id}.npy`

**Đầu ra:** `data/raw/`, `data/metadata/`, script `data/scripts/quality_control.py`.

---

### Task 2.2: Synthetic Data Generator (VAE + Physics Constraints)

**Mô tả chi tiết:**
Xây dựng bộ sinh dữ liệu tổng hợp (synthetic data generator) sử dụng Variational Autoencoders (VAE) kết hợp với các ràng buộc vật lý từ Nhánh 1 để tạo ra hàng vạn mẫu phổ giả lập nhưng tuân thủ nghiêm ngặt các định luật vật lý.

**Các bước thực hiện:**

1. **Xây dựng Physics-Constrained VAE:**
   - Kiến trúc encoder: 1D-CNN hoặc Transformer encoder → latent space z (dim = 32–128).
   - Kiến trúc decoder: Mirror architecture → reconstructed spectrum.
   - Loss function VAE:
     $\mathcal{L}_{VAE} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x) \| p(z)) + \lambda_{phys} \cdot \mathcal{L}_{physics}$
   - Trong đó $\mathcal{L}_{physics}$ là tổng các penalty constraints từ Nhánh 1:
     - Penalty nếu phổ sinh ra có đỉnh không fit Lorentzian/Voigt.
     - Penalty nếu phổ sinh ra có cường độ âm.
     - Penalty nếu phổ sinh ra có đỉnh ở vị trí không tương ứng với bất kỳ liên kết hóa học nào trong symbolic database.

2. **Geometric Transformations cho Data Augmentation:**
   - Dịch phổ theo trục X (wavenumber shift ±2 cm⁻¹): mô phỏng sai số hiệu chuẩn thiết bị.
   - Co giãn cường độ (intensity scaling ×0.8–1.2): mô phỏng khác biệt điều kiện đo.
   - Thêm nhiễu Poisson (phù hợp vật lý hơn Gaussian cho photon counting detector).
   - Thêm fluorescence background giả (polynomial bậc 3–5).
   - Mỗi phép biến đổi phải giữ nguyên ý nghĩa hóa học của phổ.

3. **Validation của Synthetic Data:**
   - So sánh phân bố thống kê (distributional comparison) giữa dữ liệu thật và synthetic bằng Fréchet Inception Distance (FID) biến thể cho 1D.
   - Kiểm tra xem một chuyên gia quang phổ có thể phân biệt synthetic vs real hay không (blind test).
   - Xác minh rằng tất cả phổ synthetic đều pass 100% physics constraints.

**Đầu ra:** `src/models/vae_generator.py`, `data/synthetic/`, notebook `notebooks/synthetic_validation.ipynb`.

---

### Task 2.3: Hệ thống chỉ số đánh giá (Physics-Informed Metrics)

**Mô tả chi tiết:**
Định nghĩa và mã hóa một bộ chỉ số đánh giá toàn diện, bao gồm cả metrics ML tiêu chuẩn và metrics chuyên biệt cho quang phổ.

**Các bước thực hiện:**

1. **Metrics ML tiêu chuẩn (để so sánh công bằng):**
   - Accuracy, Precision, Recall, F1-Score (macro/micro/weighted).
   - AUC-ROC cho bài toán phân loại.
   - MAE, RMSE cho bài toán hồi quy (ước lượng nồng độ).
   - Confusion Matrix với phân tích lỗi chi tiết.

2. **Physics-Consistency Metrics (mới):**
   - **Peak Fidelity Score (PFS):** Đo mức độ các đỉnh phổ dự đoán/phát hiện tuân theo profil Lorentzian/Voigt.
     $PFS = 1 - \frac{1}{N} \sum_{i=1}^{N} \min_{\theta} \| p_i(\nu) - V(\nu; \theta) \|_2$
     Trong đó $p_i(\nu)$ là đỉnh thứ $i$ phát hiện được, $V(\nu; \theta)$ là hàm Voigt best-fit.
   - **Chemical Consistency Score (CCS):** Tỷ lệ các đỉnh phổ được model phát hiện mà có tương ứng trong symbolic database wavenumber→bond.
   - **Energy Conservation Metric:** Kiểm tra tổng diện tích phổ trước và sau xử lý có bảo toàn không (với tolerance cho phép).

3. **Robustness Metrics:**
   - **SNR Degradation Curve:** Đo hiệu năng model ở các mức SNR khác nhau (SNR = 100, 50, 20, 10, 5, 2). Vẽ đường cong "hiệu năng vs SNR" — model Physics-Informed phải suy giảm chậm hơn model black-box.
   - **Adversarial Robustness Score:** Tỷ lệ dự đoán đúng khi thêm perturbation có mục đích (từ Task 1.2).
   - **Calibration Score:** Kiểm tra xem confidence score của model có phản ánh đúng xác suất đúng thực tế hay không (Expected Calibration Error — ECE).

4. **Reproducibility Metrics:**
   - Đo variance giữa các lần chạy khác nhau (5 random seeds).
   - Đo sự ổn định khi thay đổi nhỏ trong data split (Monte Carlo cross-validation).

**Đầu ra:** `benchmark/metrics.py`, `benchmark/evaluation.py`, `docs/metrics_specification.md`.

---

## NHÁNH 3: THE CORE ENGINE — Cấu trúc mô hình tích hợp

**Mục tiêu:** Xây dựng kiến trúc model chính — trái tim của hệ thống — kết hợp Neural Operators, Graph Neural Networks, và Physics-Informed Learning.

**Timeline dự kiến:** Tuần 5–9 (bắt đầu sau khi có đầu ra từ Nhánh 1 và 2)

**Điều kiện tiên quyết:**
- Nhánh 1: Thư viện phương trình vật lý (`engine/`) phải hoàn tất.
- Nhánh 2: Ít nhất một phiên bản data pipeline và metrics phải sẵn sàng.

### Task 3.1: Differentiable Physics Pre-processor

**Mô tả chi tiết:**
Thay vì tiền xử lý phổ bằng các bước rời rạc (discrete steps) ngoài model, tích hợp toàn bộ pipeline tiền xử lý vào đồ thị tính toán (computational graph) của mạng nơ-ron, cho phép gradient lan truyền ngược qua cả bước tiền xử lý.

**Các bước thực hiện:**

1. **Differentiable Baseline Correction:**
   - Mã hóa lại thuật toán airPLS (asymmetric Least Squares) dưới dạng differentiable:
     - Input: raw spectrum $s(\nu)$
     - Tham số học được: smoothness parameter $\lambda$, asymmetry parameter $p$
     - Output: corrected spectrum $s_{corr}(\nu) = s(\nu) - baseline(\nu; \lambda, p)$
   - Cả $\lambda$ và $p$ đều là learnable parameters — model tự học cách chọn baseline correction tối ưu.

2. **Differentiable Smoothing:**
   - Thay Savitzky-Golay filter (không differentiable do window cố định) bằng:
     - Learnable 1D convolution filter: kernel size và weights được học từ data.
     - Hoặc Gaussian smoothing với $\sigma$ là learnable parameter.
   - Constraint: Filter phải low-pass (không được khuếch đại high-frequency noise) — enforce bằng spectral constraint trên Fourier transform của filter.

3. **Differentiable Normalization:**
   - Vector normalization: $s_{norm}(\nu) = \frac{s(\nu)}{\|s\|_2}$ — đã differentiable sẵn.
   - Min-max normalization: sử dụng soft-min/soft-max để đảm bảo differentiability.
   - SNV (Standard Normal Variate): differentiable qua mean và std operations.

4. **Cosmic Ray Detection (Differentiable):**
   - Thay thuật toán spike detection cứng bằng:
     - Learnable attention mask: model học cách xác định và giảm trọng số (soft mask) các điểm bất thường thay vì loại bỏ cứng.
     - Output: element-wise weight mask $w(\nu) \in [0,1]$ applied to spectrum.

5. **Tích hợp thành Pre-processor Module:**
   - Class `DifferentiablePreprocessor(nn.Module)` chứa tất cả các bước trên.
   - Forward pass: raw spectrum → cosmic ray soft mask → baseline correction → smoothing → normalization → clean spectrum.
   - Tất cả parameters đều trainable end-to-end.

**Đầu ra:** `src/preprocessing/diff_baseline.py`, `src/preprocessing/diff_smoother.py`, `src/preprocessing/diff_normalizer.py`, `src/preprocessing/preprocessor.py` (module tích hợp).

---

### Task 3.2: Kiến trúc cốt lõi (Hybrid Neural Architecture)

**Mô tả chi tiết:**
Xây dựng kiến trúc mạng chính, kết hợp Neural Operators (cho biểu diễn liên tục), Equivariant Graph Neural Networks (cho cấu trúc phân tử), và cơ chế attention Physics-Informed.

**Các bước thực hiện:**

1. **Fourier Neural Operator (FNO) Backbone:**
   - Mục đích: Học ánh xạ từ không gian phổ sang không gian tính chất hóa lý dưới dạng toán tử liên tục (continuous operator) — không phụ thuộc vào độ phân giải (resolution-invariant).
   - Kiến trúc FNO cho 1D spectral data:
     - Lifting layer: $\mathbb{R}^1 \rightarrow \mathbb{R}^d$ (nâng chiều phổ lên feature space)
     - N Fourier layers (N = 4–8):
       Mỗi layer: $v_{t+1}(x) = \sigma\left(W \cdot v_t(x) + \mathcal{F}^{-1}(R \cdot \mathcal{F}(v_t))(x)\right)$
       trong đó $\mathcal{F}$ là FFT, $R$ là learnable filter trong Fourier space, $W$ là linear transform.
     - Projection layer: $\mathbb{R}^d \rightarrow \mathbb{R}^{out}$
   - Ưu điểm: Model có thể xử lý phổ với bất kỳ số điểm nào trên trục X (512, 1024, 2048 điểm) mà không cần retrain.

2. **Equivariant Graph Neural Network (EGNN) Module (Tùy chọn mở rộng):**
   - Mục đích: Ánh xạ các đặc trưng quang phổ (trích xuất từ FNO) vào không gian cấu trúc phân tử (molecular graph).
   - Kiến trúc:
     - Input: Feature vector từ FNO output + molecular graph structure (nếu biết cấu trúc phân tử mục tiêu).
     - Message passing layers equivariant dưới phép quay và tịnh tiến trong E(3):
       $m_{ij} = \phi_m(h_i, h_j, \|x_i - x_j\|^2, a_{ij})$
       $h_i' = \phi_h\left(h_i, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$
     - Output: Graph-level embedding qua readout function (sum/mean pooling).
   - Ứng dụng: Khi model cần dự đoán liên kết giữa phổ Raman và cấu trúc 3D phân tử.

3. **Physics-Informed Attention Mechanism:**
   - Thay vì self-attention tiêu chuẩn (Q, K, V từ data), thêm prior knowledge vào attention weights:
     - **Spectral Locality Bias:** Các wavenumber gần nhau có thể attend mạnh hơn (vì dao động phân tử gần nhau trên trục năng lượng thường liên quan).
     - **Chemical Group Attention:** Sử dụng symbolic database (từ Nhánh 1) để tạo attention mask: nếu hai wavenumber đều thuộc cùng nhóm chức (ví dụ: cả hai liên quan đến liên kết C-H), attention weight giữa chúng được boost.
   - Cơ chế: $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M_{phys}\right) V$
     trong đó $M_{phys}$ là physics-informed attention bias matrix.

4. **Multi-task Head:**
   - Từ feature representation chung, model có thể thực hiện nhiều tác vụ đồng thời:
     - Classification head: Phân loại mẫu (ung thư vs bình thường, loại vi khuẩn, loại polymer...).
     - Regression head: Ước lượng nồng độ các thành phần.
     - Peak Detection head: Xác định vị trí và cường độ các đỉnh phổ.
     - Denoising head: Khử nhiễu phổ.
   - Multi-task loss: $\mathcal{L}_{total} = \sum_{task} w_{task} \cdot \mathcal{L}_{task}$ với $w_{task}$ có thể là learnable (uncertainty weighting).

**Đầu ra:** `src/models/fno_backbone.py`, `src/models/egnn_module.py`, `src/models/physics_attention.py`, `src/models/multi_task_heads.py`, `src/models/hybrid_model.py` (tích hợp).

---

### Task 3.3: Physics-Preserving Loss Function

**Mô tả chi tiết:**
Thiết kế hàm loss tổng hợp (composite loss function) ép model tuân thủ vật lý trong quá trình học.

**Các bước thực hiện:**

1. **Thành phần Data-Driven Loss:**
   - Classification: Cross-Entropy Loss hoặc Focal Loss (cho imbalanced data).
   - Regression: MSE / Huber Loss.
   - Peak Detection: Combination of position loss (L1 cho vị trí peak) + amplitude loss (MSE cho cường độ) + count loss (Poisson NLL cho số peak).

2. **Thành phần Physics-Preserving Penalty:**
   - **Lorentzian Fidelity Penalty:**
     $\mathcal{L}_{Lor} = \sum_{i} \min_{\theta_i} \left\| p_i^{pred}(\nu) - L(\nu; A_i, \nu_{0,i}, \Gamma_i) \right\|_2^2$
     Với $L$ là hàm Lorentzian, $\theta_i = (A_i, \nu_{0,i}, \Gamma_i)$ được fit bằng differentiable least-squares.
   - **Non-Negativity Penalty:**
     $\mathcal{L}_{nn} = \sum_{\nu} \max(0, -\hat{s}(\nu))^2$
   - **Smoothness Penalty (Tikhonov Regularization trên phổ):**
     $\mathcal{L}_{smooth} = \left\| \frac{d^2 \hat{s}}{d\nu^2} \right\|_2^2$
   - **Spectral Conservation Penalty:**
     $\mathcal{L}_{conserv} = \left| \int \hat{s}(\nu) d\nu - \int s_{input}(\nu) d\nu \right|^2$
   - **Chemical Consistency Penalty:**
     Penalty cho mỗi đỉnh phổ mà model phát hiện ở vị trí không tương ứng với bất kỳ mode dao động nào trong symbolic database.

3. **Tổng hợp Loss:**
   $\mathcal{L} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{Lor} + \lambda_2 \mathcal{L}_{nn} + \lambda_3 \mathcal{L}_{smooth} + \lambda_4 \mathcal{L}_{conserv} + \lambda_5 \mathcal{L}_{chem}$
   - Các hệ số $\lambda_i$ có thể:
     - Cố định (handcrafted) dựa trên validation.
     - Hoặc adaptive (learnable) sử dụng phương pháp GradNorm hoặc uncertainty weighting.

4. **Curriculum Learning cho Physics Loss:**
   - Giai đoạn 1 (warm-up): Chỉ train $\mathcal{L}_{data}$ để model có khả năng cơ bản.
   - Giai đoạn 2: Dần tăng $\lambda_i$ theo schedule (linear hoặc exponential) để model từ từ "học" tuân thủ vật lý.
   - Lý do: Nếu enforce physics quá sớm khi model chưa có biểu diễn tốt, gradient từ physics loss có thể gây instability.

**Đầu ra:** `src/training/physics_loss.py`, `src/training/loss_scheduler.py`.

---

### Task 3.4: Training Pipeline

**Mô tả chi tiết:**
Xây dựng pipeline huấn luyện end-to-end có logging, checkpointing, và giám sát physics compliance trong suốt quá trình training.

**Các bước thực hiện:**

1. **Data Loading:**
   - DataLoader hỗ trợ cả real và synthetic data với tỷ lệ trộn (mixing ratio) có thể cấu hình.
   - On-the-fly augmentation sử dụng geometric transformations (từ Task 2.2).
   - Stratified splitting: train/val/test đảm bảo cân bằng nhãn.

2. **Training Loop:**
   - Optimizer: AdamW hoặc LAMB (cho large batch training).
   - Learning rate schedule: Cosine annealing with warm restarts.
   - Gradient clipping: max_norm = 1.0 (ngăn gradient explosion từ physics loss).
   - Mixed-precision training (FP16/BF16) nếu phần cứng hỗ trợ.

3. **Monitoring & Logging:**
   - Log mỗi epoch: tất cả thành phần loss (data loss, từng physics penalty riêng biệt), tất cả metrics (ML + physics-consistency).
   - Sử dụng Weights & Biases (wandb) hoặc TensorBoard.
   - Physics Compliance Dashboard: real-time tracking tỷ lệ phổ output vi phạm từng constraint.

4. **Checkpointing:**
   - Lưu checkpoint tốt nhất dựa trên composite score: $S = w_{acc} \cdot \text{F1} + w_{phys} \cdot \text{PFS} + w_{rob} \cdot \text{Robustness}$
   - Lưu cả early stopping checkpoint.

5. **Hyperparameter Search:**
   - Sử dụng Optuna hoặc Ray Tune.
   - Các hyperparameters cần search: learning rate, batch size, FNO depth, latent dim, các hệ số $\lambda_i$, mixing ratio real/synthetic.

**Đầu ra:** `src/training/trainer.py`, `src/training/dataloader.py`, `configs.yaml`, `src/training/hparam_search.py`.

---

## NHÁNH 4: THE INTERFACE — Module giải thích & Tự động hóa

**Mục tiêu:** Đảm bảo mọi kết quả từ model đều giải thích được bằng ngôn ngữ hóa lý, và xây dựng pipeline tự động hóa end-to-end cho người dùng cuối.

**Timeline dự kiến:** Tuần 7–12 (bắt đầu khi Nhánh 3 có model huấn luyện sơ bộ)

**Điều kiện tiên quyết:** Nhánh 3 phải có model checkpoint có thể chạy inference.

### Task 4.1: Explainable AI (XAI) Module — Physics-Grounded Explanations

**Mô tả chi tiết:**
Xây dựng module XAI không chỉ hiện heatmap mà phải trả lời được câu hỏi: "Model dự đoán A vì lý do vật lý/hóa học gì?"

**Các bước thực hiện:**

1. **Spectral Attribution Analysis:**
   - Tích hợp Integrated Gradients hoặc SHAP cho phổ 1D:
     - Output: Attribution map $a(\nu)$ — mức đóng góp của mỗi wavenumber vào quyết định phân loại.
   - Post-processing attribution: group các wavenumber liền kề có attribution cao thành "attribution regions", sau đó map mỗi region vào symbolic database.

2. **Symbolic Explanation Generator:**
   - Input: Attribution map + Symbolic database.
   - Thuật toán:
     a. Xác định top-K attribution regions (wavenumber ranges có tổng attribution cao nhất).
     b. Tra cứu mỗi region trong symbolic database → Xác định liên kết hóa học / nhóm chức tương ứng.
     c. So sánh cường độ/vị trí đỉnh trong region đó với giá trị tham chiếu (reference values) trong cơ sở dữ liệu.
     d. Sinh câu giải thích dạng template:
        - "Phân loại mẫu là [Label] với độ tin cậy [Confidence]% vì:
          (1) Cường độ đỉnh tại [Wavenumber] cm⁻¹ (liên kết [Bond Type]) [tăng/giảm] [X]% so với tham chiếu, cho thấy [Hóa học giải thích].
          (2) Tỷ lệ cường độ giữa [Peak A] và [Peak B] = [Value], nhất quán với [Phương trình/Rule]."

3. **Uncertainty Quantification:**
   - Monte Carlo Dropout: chạy inference N lần (N = 50–100) với dropout active → ước lượng phân bố dự đoán.
   - Hoặc Deep Ensemble: train 5 models với random seed khác nhau → lấy mean và std.
   - Output: $\hat{y} \pm \delta y$ với $\delta y$ là epistemic uncertainty.
   - Thêm aleatoric uncertainty estimation từ heteroscedastic output head.

4. **Confidence Calibration:**
   - Áp dụng Temperature Scaling hoặc Platt Scaling trên validation set.
   - Kiểm tra ECE (Expected Calibration Error) trước và sau calibration.

**Đầu ra:** `interface/xai_explainer.py`, `interface/symbolic_extract.py`, `interface/uncertainty.py`.

---

### Task 4.2: Task Automation Engine — End-to-End Pipeline

**Mô tả chi tiết:**
Xây dựng pipeline tự động hóa hoàn chỉnh: người dùng chỉ cần đưa vào 1 file phổ Raman thô, hệ thống tự động thực hiện toàn bộ quy trình phân tích.

**Các bước thực hiện:**

1. **Input Handler:**
   - Hỗ trợ nhiều format đầu vào: `.csv`, `.txt`, `.spc`, `.wdf` (Renishaw), `.npy`.
   - Auto-detect: trục X (cm⁻¹ hay nm), separator, header rows.
   - Validation: kiểm tra range wavenumber hợp lý, kiểm tra dữ liệu thiếu.

2. **Automated Analysis Pipeline:**
   - Bước 1: Differentiable Pre-processing (từ Task 3.1) — tự động khử nhiễu nền, loại cosmic ray, chuẩn hóa.
   - Bước 2: Peak Detection & Identification — phát hiện tất cả đỉnh phổ, fit Voigt profile, trả về vị trí + cường độ + FWHM (Full Width at Half Maximum).
   - Bước 3: Chemical Database Matching — so khớp các đỉnh phát hiện được với cơ sở dữ liệu hóa chất (symbolic database + RRUFF + thư viện nội bộ) → Trả về danh sách các hợp chất / liên kết hóa học ứng viên.
   - Bước 4: Classification/Regression — chạy model chính (từ Task 3.2) → Trả về nhãn/giá trị dự đoán + uncertainty.
   - Bước 5: XAI Explanation — sinh giải thích symbolic (từ Task 4.1).
   - Bước 6: Report Generation — tổng hợp tất cả kết quả thành báo cáo.

3. **Report Generator:**
   - Tự động sinh báo cáo PDF bao gồm:
     - Phổ gốc và phổ sau xử lý (plot chồng lên nhau).
     - Bảng đỉnh phổ phát hiện được (vị trí, cường độ, assignment hóa học).
     - Kết quả phân loại/hồi quy + uncertainty.
     - Giải thích symbolic.
     - Phổ tham chiếu (reference spectra) từ database matching.
   - Format: PDF (sử dụng reportlab hoặc matplotlib + FPDF) hoặc HTML report.

**Đầu ra:** `src/inference/pipeline.py`, `src/inference/input_handler.py`, `src/inference/report_generator.py`.

---

### Task 4.3: Dashboard (Web Interface)

**Mô tả chi tiết:**
Xây dựng giao diện web cho phép người dùng không chuyên tin học có thể sử dụng toàn bộ hệ thống.

**Các bước thực hiện:**

1. **Framework:** Streamlit hoặc Gradio (ưu tiên Gradio cho deployment nhanh).

2. **Chức năng chính:**
   - Upload file phổ (kéo-thả hoặc chọn file).
   - Hiển thị phổ gốc (interactive plot với zoom, pan).
   - Chọn tác vụ: Phân loại / Ước lượng nồng độ / Peak analysis / Khử nhiễu.
   - Nút "Analyze" → chạy full pipeline → hiển thị kết quả.
   - Tab kết quả: phổ đã xử lý, bảng peaks, chemical assignments, classification result, explanation text, uncertainty bar.
   - Tải báo cáo PDF.

3. **Batch Processing:**
   - Upload nhiều file cùng lúc → xử lý song song → tải kết quả dạng ZIP.

4. **Tùy chỉnh nâng cao (cho chuyên gia):**
   - Chọn/bỏ các bước tiền xử lý.
   - Điều chỉnh threshold cho peak detection.
   - Chọn model/checkpoint cụ thể.
   - Hiển thị thêm: SHAP attribution map, uncertainty distribution.

**Đầu ra:** `dashboard/app.py`, `dashboard/assets/`.

---

## GIAI ĐOẠN HỘI TỤ: INTEGRATION & VALIDATION

**Timeline dự kiến:** Tuần 10–12 (chồng lấp với cuối Nhánh 4)

### Task INT.1: Integration Testing

**Mô tả chi tiết:**
Tích hợp tất cả module từ 4 nhánh thành một pipeline liền mạch và kiểm tra end-to-end.

**Các bước thực hiện:**

1. **Smoke Test:** Chạy pipeline hoàn chỉnh với 10 phổ thử → kiểm tra không có crash, output format đúng.
2. **Physics Integrity Test:** Chạy `tests/test_physics.py` — kiểm tra tất cả physics constraints hoạt động đúng.
3. **Regression Test:** So sánh kết quả model với baseline (từ Task 1.1) — model mới phải ≥ baseline trên mọi metrics.
4. **Edge Case Test:** Phổ toàn nhiễu, phổ phẳng (không có peak), phổ saturated, phổ với cosmic ray chưa loại → model không được crash và phải trả về uncertainty cao.

### Task INT.2: Ablation Study

**Mô tả chi tiết:**
Chứng minh đóng góp của từng thành phần Physics-Informed bằng cách lần lượt bỏ chúng ra.

**Các bước thực hiện:**

1. **Full model** (tất cả physics components) → Ghi nhận hiệu năng.
2. **Bỏ Physics Loss** → Chỉ train bằng data loss → Ghi nhận suy giảm.
3. **Bỏ Differentiable Preprocessing** → Dùng preprocessing cố định → Ghi nhận.
4. **Bỏ Physics Attention** → Dùng standard attention → Ghi nhận.
5. **Bỏ Symbolic Constraint** → Không penalty vị trí peak → Ghi nhận.
6. **Black Box model (1D-CNN thuần)** → So sánh toàn diện.
7. Tổng hợp thành bảng ablation study.

### Task INT.3: Final Evaluation & Reporting

1. **Chạy evaluation cuối cùng trên test set** (chưa từng thấy trong quá trình phát triển).
2. **Tổng hợp tất cả metrics** vào bảng so sánh cuối cùng.
3. **Sinh báo cáo dự án** (`results/report.pdf`): kết quả, ablation, so sánh baseline, thảo luận.

---

## ĐẦU RA: CẤU TRÚC THƯ MỤC DỰ ÁN

```
Raman_Physics_Informed_AI/
│
├── /docs                          # Tài liệu khoa học
│   ├── groundwork.md              # Chứng minh lỗ hổng Black Box + Framework toán học
│   ├── sota_review.md             # SOTA review có hệ thống
│   ├── metrics_specification.md   # Định nghĩa chi tiết tất cả metrics
│   ├── model_architecture.md      # Mô tả kiến trúc model
│   └── api_reference.md           # API documentation cho code
│
├── /data                          # Dữ liệu
│   ├── /raw                       # Phổ Raman gốc (theo dataset name)
│   ├── /synthetic                 # Phổ sinh từ VAE generator
│   ├── /processed                 # Phổ đã qua quality control
│   ├── /metadata                  # File nhãn, thông số mẫu, provenance
│   └── /scripts                   # Scripts thu thập và xử lý data
│       └── quality_control.py
│
├── /engine                        # Module kiến thức lý thuyết (Nhánh 1)
│   ├── physics_laws.py            # Hàm differentiable: Lorentzian, Voigt, Rayleigh
│   ├── symbolic_math.py           # Mapping wavenumber → chemical bonds
│   ├── constraints.py             # Hàm penalty vật lý cho loss function
│   └── spectral_database.yaml    # Cơ sở dữ liệu symbolic knowledge
│
├── /src                           # Mã nguồn cốt lõi (Nhánh 3)
│   ├── /preprocessing             # Differentiable preprocessing
│   │   ├── diff_baseline.py       # Differentiable airPLS
│   │   ├── diff_smoother.py       # Learnable smoothing filter
│   │   ├── diff_normalizer.py     # Differentiable normalization
│   │   └── preprocessor.py        # Tích hợp tất cả preprocessing
│   │
│   ├── /models                    # Kiến trúc model
│   │   ├── fno_backbone.py        # Fourier Neural Operator
│   │   ├── egnn_module.py         # Equivariant Graph Neural Network
│   │   ├── physics_attention.py   # Physics-Informed Attention
│   │   ├── multi_task_heads.py    # Classification/Regression/Peak heads
│   │   ├── hybrid_model.py        # Model tích hợp hoàn chỉnh
│   │   └── vae_generator.py       # VAE cho synthetic data (Nhánh 2)
│   │
│   ├── /training                  # Pipeline huấn luyện
│   │   ├── trainer.py             # Training loop chính
│   │   ├── dataloader.py          # Data loading + augmentation
│   │   ├── physics_loss.py        # Composite physics-preserving loss
│   │   ├── loss_scheduler.py      # Curriculum learning scheduler
│   │   └── hparam_search.py       # Hyperparameter optimization
│   │
│   └── /inference                 # Pipeline suy luận
│       ├── pipeline.py            # End-to-end inference pipeline
│       ├── input_handler.py       # Multi-format input parser
│       └── report_generator.py    # Tự động sinh báo cáo PDF/HTML
│
├── /benchmark                     # Chuẩn đánh giá (Nhánh 2)
│   ├── metrics.py                 # ML metrics + Physics-Consistency metrics
│   ├── evaluation.py              # Script so sánh toàn diện
│   └── /baselines                 # Code reproduce các baseline
│       ├── pca_svm.py
│       ├── pls_da.py
│       └── cnn_baseline.py
│
├── /interface                     # Module XAI & Symbolic AI (Nhánh 4)
│   ├── xai_explainer.py           # Spectral attribution + symbolic explanation
│   ├── symbolic_extract.py        # Trích xuất phương trình giải thích
│   └── uncertainty.py             # MC Dropout / Deep Ensemble uncertainty
│
├── /dashboard                     # Giao diện web
│   ├── app.py                     # Streamlit/Gradio web app
│   └── /assets                    # Static files (CSS, icons)
│
├── /results                       # Kết quả & sản phẩm
│   ├── /checkpoints               # Trọng số model (best, last, ablation variants)
│   ├── /figures                   # Plots, đồ thị loss, phổ dự đoán
│   ├── /ablation                  # Kết quả ablation study
│   └── report.pdf                 # Báo cáo khoa học tổng hợp
│
├── /tests                         # Unit tests & Integration tests
│   ├── test_physics.py            # Test physics constraints
│   ├── test_preprocessing.py      # Test differentiable preprocessing
│   ├── test_model.py              # Test model forward/backward pass
│   ├── test_pipeline.py           # Test end-to-end inference
│   └── test_edge_cases.py         # Test các trường hợp đặc biệt
│
├── /notebooks                     # Jupyter notebooks cho exploration
│   ├── adversarial_analysis.ipynb # Phân tích Black Box (Task 1.2)
│   ├── synthetic_validation.ipynb # Validation synthetic data (Task 2.2)
│   ├── training_analysis.ipynb    # Phân tích quá trình training
│   └── demo.ipynb                 # Demo notebook cho showcase
│
├── configs.yaml                   # Cấu hình siêu tham số, đường dẫn, experiment config
├── Dockerfile                     # Container cho reproducibility
├── Makefile                       # Lệnh tự động: make train, make test, make dashboard, make report
├── requirements.txt               # Dependencies (PyTorch, JAX, neuraloperator, DGL, scikit-learn...)
├── setup.py                       # Package setup
├── LICENSE                        # Giấy phép
└── README.md                      # Hướng dẫn tổng quan dự án
```

---

## TIMELINE TỔNG HỢP (12 Tuần)

| Tuần | Nhánh 1 | Nhánh 2 | Nhánh 3 | Nhánh 4 | Integration |
|------|---------|---------|---------|---------|-------------|
| 1    | Task 1.1: SOTA Review | Task 2.1: Data collection | — | — | — |
| 2    | Task 1.1 (tiếp) | Task 2.1 (tiếp) | — | — | — |
| 3    | Task 1.2: Black Box proof | Task 2.2: VAE Generator | — | — | — |
| 4    | Task 1.3: Math framework | Task 2.2 + 2.3: Metrics | — | — | — |
| 5    | — | Task 2.3 (hoàn tất) | Task 3.1: Diff. Preproc. | — | — |
| 6    | — | — | Task 3.2: Architecture | — | — |
| 7    | — | — | Task 3.2 (tiếp) + 3.3: Loss | Task 4.1: XAI (bắt đầu) | — |
| 8    | — | — | Task 3.4: Training | Task 4.1 (tiếp) | — |
| 9    | — | — | Task 3.4 (hoàn tất) | Task 4.2: Automation | — |
| 10   | — | — | — | Task 4.2 + 4.3: Dashboard | INT.1: Integration |
| 11   | — | — | — | Task 4.3 (hoàn tất) | INT.2: Ablation |
| 12   | — | — | — | — | INT.3: Final eval + Report |

---

## PHỤ LỤC: DANH SÁCH DEPENDENCIES CHÍNH

**Core ML:**
- PyTorch ≥ 2.0 (hoặc JAX + Flax nếu ưu tiên differentiable programming)
- neuraloperator (thư viện chính thức cho FNO)
- torch-geometric hoặc DGL (cho EGNN)
- scikit-learn (cho baselines)

**Spectroscopy-specific:**
- RamanSPy (thư viện phân tích Raman cho Python)
- scipy.signal (Savitzky-Golay, peak finding)
- lmfit (curve fitting — Voigt, Lorentzian)

**XAI & Evaluation:**
- captum (PyTorch XAI: Integrated Gradients, SHAP)
- uncertainty-toolbox (calibration metrics)

**Data & Visualization:**
- numpy, pandas
- matplotlib, plotly (interactive plots)
- wandb hoặc tensorboard

**Deployment:**
- streamlit hoặc gradio
- reportlab (PDF generation)
- docker

**Hyperparameter Optimization:**
- optuna hoặc ray[tune]

---

## GHI CHÚ QUAN TRỌNG

1. **Tính tái tạo (Reproducibility):** Mọi thí nghiệm phải ghi lại random seed, hardware specs, exact library versions (lock trong `requirements.txt`). Dockerfile đảm bảo bất kỳ ai cũng có thể reproduce kết quả.

2. **Version Control:** Sử dụng Git với conventional commits. Mỗi nhánh workflow tương ứng với một Git branch (`feature/groundwork`, `feature/gold-standard`, `feature/core-engine`, `feature/interface`), merge vào `develop` khi hoàn tất, và `main` cho release cuối cùng.

3. **Mở rộng IR:** Kiến trúc được thiết kế modular — khi mở rộng sang phổ IR, chỉ cần: (a) thêm phương trình Beer-Lambert vào `engine/`, (b) thêm IR symbolic database, (c) retrain hoặc fine-tune model. Không cần thay đổi kiến trúc cốt lõi.

4. **Nguyên tắc "Physics First":** Trong mọi quyết định thiết kế, nếu có xung đột giữa hiệu năng ML thuần túy và tính nhất quán vật lý, luôn ưu tiên vật lý. Một model có accuracy 95% nhưng 100% physics-consistent tốt hơn model 98% accuracy nhưng vi phạm vật lý ở 5% trường hợp.