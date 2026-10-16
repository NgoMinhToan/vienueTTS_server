# Kế Hoạch Chuẩn Hóa & Kiểm Thử vienueTTS_server

**Mục tiêu:** Xây dựng `vienueTTS_server` thành một AI Inference Microservice độc lập 100%, stateless, hiệu năng cao, có khả năng tự phục vụ (tự tải model hoặc nạp từ volume) và cung cấp đầy đủ thông tin giọng nói cho bất kỳ client nào kết nối tới.

---

## 1. Trạng Thái Hiện Tại & Đánh Giá
- **Ưu điểm:**
  - Đã tích hợp VieNeu-TTS v3 Turbo/Nano chạy ONNX Runtime trên CPU (torch-free).
  - Tích hợp `huggingface_hub`, `tokenizers`, `sea-g2p`, `kaldi-native-fbank`, `soxr`, `vieneu>=3.8.3` trong `requirements.txt`.
  - Hỗ trợ xuất âm thanh chuẩn 48kHz WAV và PCM.
  - Hỗ trợ trích xuất speaker embedding `.npz` qua `/embed`.
  - Hỗ trợ nạp embedding clone qua `ref_npz_base64` hoặc `ref_npz_path`.
  - Hỗ trợ Private Network Access (PNA) middleware cho web client.
- **Vấn đề đã khắc phục:**
  - Đã bỏ cờ cưỡng chế `HF_HUB_OFFLINE: 1` và cờ `:ro`, cho phép server tự động tải model từ Hugging Face Hub (`pnnbao-ump/VieNeu-TTS-v3-Turbo` và `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX`) khi triển khai trên VPS mới mà không cần copy thủ công.
  - Bổ sung endpoint danh mục 23 giọng nói `/voices`.

---

## 2. Danh Sách API Contract Chuẩn Hóa

| Phương thức | Endpoint | Xác thực | Mô tả |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` / `/internal/v1/health` | Không | Trạng thái server, số lượng giọng khả dụng (`total_voices: 23`), trạng thái nạp engine, active jobs |
| `GET` | `/internal/v1/voices` / `/v1/voices` | `X-API-Key` | Danh sách 23 giọng preset với metadata (id, name, gender, description, style, 48kHz) |
| `GET` | `/internal/v1/voices/{id}` | `X-API-Key` | Chi tiết một giọng theo ID hoặc Tên (ví dụ: `minh_duc`, `quynh_anh`, `Minh Đức`) |
| `POST` | `/internal/v1/synthesize` / `/v1/synthesize` | `X-API-Key` | Tổng hợp văn bản thành 48kHz WAV audio. Hỗ trợ giọng preset hoặc giọng clone (`ref_npz_base64`, `ref_npz_path`, `ref_audio_path`) |
| `POST` | `/internal/v1/embed` / `/v1/embed` | `X-API-Key` | Trích xuất speaker embedding từ file audio mẫu sang binary NumPy `.npz` |
| `GET` | `/internal/v1/metrics` | `X-API-Key` | Telemetry độ trễ trung bình, tổng số request, active jobs |

---

## 3. Quản Lý Trọng Số Model (Model Weights Management)
1. **Chế độ Tự Động Tải (Zero-Setup / First-Run trên VPS):**
   - Khi thư mục `./models` trống hoặc không mount volume, server tự động tải các file graph ONNX và Codec từ Hugging Face Hub về `HF_HOME`.
   - `HF_HUB_OFFLINE` mặc định là `0`.
2. **Chế độ Offline / Đã Có Sẵn Model:**
   - Nếu mount thư mục đã có sẵn model (ví dụ: `data/models`), server nhận diện file trong cache và chạy offline ngay lập tức, không tốn băng thông internet.

---

## 4. Hệ Thống Logging Chuẩn Hóa (Docker & CLI Visibility)
- **Banner Khởi Động:** Hiển thị trực quan thông tin Engine Mode, Audio Standard (48,000 Hz Studio WAV), Intra-op Threads, Checkpoints directory, Available voices count, HTTP Endpoint, và API Key protection.
- **Inference Telemetry Logging:** Mỗi request tổng hợp âm thanh in rõ ràng:
  - Text snippet, voice identifier (preset hoặc cloned npz), speed, pitch, emotion, engine mode.
  - Sau khi hoàn thành: Thời gian thực thi (elapsed s), độ dài âm thanh (audio_dur @ 48kHz), số lượng samples, và hệ số thời gian thực RTF (Real-Time Factor).
- **Voice Enrollment Logging:** In thông tin file âm thanh tham chiếu và thời gian trích xuất embedding `.npz`.

---

## 5. Kế Hoạch Kiểm Thử (Verification Suite)
- [x] Test health check mở không cần key (`test_health_check_open_without_api_key`, `test_root_health_open`).
- [x] Test bảo vệ API Key (`test_voices_list_rejected_without_api_key`, `test_synthesize_rejected_without_api_key`).
- [x] Test danh sách 23 giọng nói (`test_voices_list_with_api_key`, `test_voice_detail_endpoint`).
- [x] Test tổng hợp âm thanh giọng preset (`test_synthesize_accepted_with_valid_api_key`).
- [x] Test tổng hợp giọng clone qua Base64 `.npz` (`test_synthesize_with_cloned_voice_npz_base64`).
- [x] Test trích xuất speaker embedding `.npz` (`test_extract_speaker_embedding_embed_endpoint`).
- [x] Test header Private Network Access PNA preflight (`test_pna_options_header`).
- [x] Test giá trị biên (Boundary Value Analysis): text rỗng, text vượt 5000 ký tự, speed/pitch/emotion ngoài giới hạn (`test_synthesize_speed_out_of_bounds`, `test_synthesize_pitch_out_of_bounds`, `test_synthesize_emotion_out_of_bounds`).
- [x] Test phòng vệ bảo mật: Chặn path traversal `ref_audio_path` (`../../etc/passwd`), chặn file đuôi cấm (`.exe`), chặn Base64 hỏng (`test_synthesize_path_traversal_ref_audio_rejected`, `test_synthesize_invalid_audio_extension_rejected`, `test_synthesize_invalid_base64_npz_rejected`).
- [x] Test xử lý ngoại lệ tra cứu giọng không tồn tại 404 (`test_get_non_existent_voice_detail_returns_404`).
- [x] Toàn bộ 21/21 test cases trong `vienueTTS_server/tests/` đều vượt qua 100%.
- [x] Đạt 12/12 test case trong `vienueTTS_server/tests/test_inference_auth.py`.
