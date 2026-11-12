# BÁO CÁO ĐÁNH GIÁ KỸ THUẬT: AI TECHNICAL REPORT GENERATOR

**Đơn vị/Người đánh giá:** AI Code Reviewer
**Phạm vi đánh giá:** Toàn bộ mã nguồn trong repository (`app.py`, `config.py`, `core/`, `llm/`, `models/`, `utils/`, `prompts/`, `tests/`) tại commit `7ab8f1f`
**Phương pháp:** Đọc mã nguồn tĩnh, chạy bộ kiểm thử, thực nghiệm có kiểm soát trên các hàm heuristic của validator
**Ngày đánh giá:** 12/11/2026

---

## TÓM TẮT ĐÁNH GIÁ (EXECUTIVE SUMMARY)

Hệ thống là một ứng dụng sinh báo cáo đồ án kỹ thuật bán tự động, triển khai theo mô hình **map–reduce dựa trên LLM** với một tầng tri thức trung gian có cấu trúc (`ProjectKnowledge`) và bước **human-in-the-loop** cho phép người dùng kiểm duyệt tri thức trước khi sinh văn bản. Mã nguồn có tổ chức tốt, kiểu dữ liệu rõ ràng, xử lý lỗi theo nguyên tắc "lỗi cục bộ không làm sập pipeline", và tầng nạp file có các biện pháp bảo mật archive ở mức tốt so với phạm vi một MVP.

Bộ kiểm thử gồm **35 test, toàn bộ đạt (35 passed, 3.46s)**, sử dụng `MockLLMClient` nên chạy được hoàn toàn offline.

Các điểm cần khắc phục ưu tiên cao:

| # | Vấn đề | Vị trí | Mức độ |
|---|--------|--------|--------|
| 1 | `MAX_CHUNKS` được khai báo nhưng **không được áp dụng** ở bất kỳ đâu → không có trần chi phí/số lời gọi LLM | [config.py:33](../config.py#L33), [core/pipeline.py:98](../core/pipeline.py#L98) | Cao |
| 2 | Kiểm tra feature trong validator đánh giá chính `ProjectKnowledge` chứ không phải report → sinh issue `high` mà revision **không thể sửa** | [core/validator.py:278-286](../core/validator.py#L278-L286) | Cao |
| 3 | Heuristic phát hiện công nghệ khớp từ "dùng" trong cụm "người dùng" → tỷ lệ dương tính giả cao | [core/validator.py:22-26](../core/validator.py#L22-L26) | Cao |
| 4 | Bước reduce đưa toàn bộ chunk analyses vào **một** prompt duy nhất → rủi ro vượt context window với dự án lớn | [core/knowledge_builder.py:77-94](../core/knowledge_builder.py#L77-L94) | Trung bình |
| 5 | Khoá cache chỉ dựa trên SHA-256 nội dung, không gồm model/prompt version → cache "cũ" sau khi đổi prompt hoặc model | [core/analyzer.py:79-80](../core/analyzer.py#L79-L80) | Trung bình |

> **Cập nhật sau đánh giá:** Các mục 1, 2, 3 và 5 trong bảng trên, cùng 7 khiếm khuyết khác ở mục 4.2.2, đã được khắc phục và bổ sung test hồi quy. Mục 4 (reduce một lần) thuộc phạm vi thay đổi kiến trúc nên chưa xử lý. Chi tiết xem **PHỤ LỤC A**. Chương 1–5 được giữ nguyên để phản ánh trạng thái mã nguồn tại commit `7ab8f1f`; các tham chiếu số dòng trong các chương này ứng với phiên bản đó.

---

## CHƯƠNG 1: TỔNG QUAN HỆ THỐNG VÀ LUỒNG XỬ LÝ (PIPELINE)

### 1.1. Tổng quan kiến trúc

**Bài toán:** Sinh viên thực hiện đồ án thường phải viết báo cáo kỹ thuật mô tả kiến trúc, công nghệ, module và kết quả từ mã nguồn đã viết. Việc đưa trực tiếp toàn bộ source code vào một LLM gặp ba trở ngại: (1) giới hạn context window, (2) hiện tượng hallucination — LLM tự bịa công nghệ, tính năng hoặc số liệu thực nghiệm, và (3) thiếu khả năng truy vết (traceability) giữa nhận định trong báo cáo và mã nguồn gốc.

**Giải pháp kiến trúc:** Hệ thống tách bài toán thành hai pha độc lập:

1. **Pha trích xuất tri thức (Knowledge Extraction):** Source code được chia nhỏ, mỗi chunk được LLM phân tích độc lập thành JSON có schema (`ChunkAnalysis`), sau đó được tổng hợp (reduce) thành `ProjectKnowledge` — một cấu trúc dữ liệu có kiểm chứng, kèm danh sách `Evidence` trỏ về `source_file`.
2. **Pha sinh văn bản (Report Generation):** LLM chỉ nhận `ProjectKnowledge` (không nhận raw source code) để sinh báo cáo Markdown 5 chương, sau đó được kiểm tra bởi validator kết hợp deterministic + LLM.

Giữa hai pha là bước **review thủ công** (Step 4 trên UI): người dùng có thể đọc, chỉnh sửa JSON và lưu lại sau khi qua Pydantic validation. Đây là điểm thiết kế quan trọng nhất của hệ thống: tri thức trở thành một artifact có thể kiểm duyệt, thay vì là trạng thái ẩn bên trong prompt.

**Quy mô mã nguồn:**

| Thành phần | Số dòng | Ghi chú |
|------------|--------:|---------|
| `app.py` | 407 | Giao diện Streamlit 6 bước |
| `config.py` | 85 | Cấu hình runtime |
| `core/` (9 module) | 2 304 | Lõi pipeline |
| `llm/llm_client.py` | 172 | Client OpenAI-compatible + mock |
| `models/project_schema.py` | 94 | 7 Pydantic schema |
| `utils/` (3 module) | 155 | Cache, JSON, logging |
| `tests/` (13 file) | 871 | 35 test case |
| `prompts/` (5 file) | ~7.6 KB | Prompt template tách khỏi code |

### 1.2. Sơ đồ dòng dữ liệu (Data Flow)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ app.py (Streamlit)   st.session_state["uploaded_files"] = [{filename,     │
│                      content: bytes, size}]                               │
└──────────────┬───────────────────────────────────────────────────────────┘
               │ analyze_project()  [core/pipeline.py]
               │ ① kiểm tra MAX_TOTAL_UPLOAD_SIZE
               ▼
┌──────────────────────────┐   ZIP → _load_zip_file(): path traversal check,
│ file_loader.load_files() │   member limit, bounded read, symlink/encrypted
│  → list[dict] {filename, │   reject; PDF → PyMuPDF; DOCX → python-docx;
│    status, content,      │   text → UTF-8-sig → cp1252 → replace
│    content_sha256}       │   status ∈ {loaded, skipped, error, expanded}
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐   NFC, xoá ký tự điều khiển Unicode (Cc/Cf/...),
│ preprocessor             │   chuẩn hoá newline, rstrip, gộp >2 dòng trống
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐   CharacterChunker(size=6000, overlap=500)
│ chunker.chunk_documents  │   → {filename, chunk_index, content,
└────────────┬─────────────┘      content_sha256, start_char, end_char}
             ▼
┌──────────────────────────┐   Tuần tự từng chunk:
│ analyzer (MAP)           │   cache.get(sha256) ─hit→ rebind filename
│  N lời gọi LLM           │                     ─miss→ LLM → _prepare_payload
│                          │   → ChunkAnalysis.model_validate → cache.set
└────────────┬─────────────┘   Lỗi từng chunk → AnalysisRun.errors (không dừng)
             ▼
┌──────────────────────────┐   1 lời gọi LLM với toàn bộ analyses JSON
│ knowledge_builder        │   → _apply_evidence_boundary(): giao với tập
│ (REDUCE)                 │     tri thức đã biết từ analyses
└────────────┬─────────────┘
             ▼
     ProjectKnowledge  ──►  [Step 4] Người dùng review / sửa JSON
             │                      (Pydantic validate trước khi lưu)
             ▼
┌──────────────────────────┐   1 lời gọi LLM, chỉ nhận ProjectKnowledge JSON
│ report_generator         │   normalize_report_markdown(): bổ sung tiêu đề,
└────────────┬─────────────┘   chèn chương thiếu với "Không đủ dữ liệu..."
             ▼
┌──────────────────────────┐   validate_code (regex) + validate_llm
│ validator                │   while có issue "high" và revision < MAX_REVISION:
│                          │       revise → guard → validate lại
└────────────┬─────────────┘
             ▼
     exporter: technical_report_<slug>.md  (download_button)
```

**Số lời gọi LLM cho một lần chạy đầy đủ:** `N_chunks (map, trừ cache hit) + 1 (reduce) + 1 (generate) + (1 + MAX_REVISION) × 1 (LLM validate) + ≤ MAX_REVISION (revise)`. Với cấu hình mặc định `MAX_REVISION=1`, tổng là `N + 4` đến `N + 5` lời gọi.

---

## CHƯƠNG 2: ĐÁNH GIÁ THIẾT KẾ MODULE VÀ CÔNG NGHỆ

### 2.1. Phân tích các thành phần cốt lõi

| Module | Trách nhiệm | Điểm mạnh | Điểm cần cải thiện |
|--------|-------------|-----------|--------------------|
| `app.py` | UI 6 bước, quản lý `st.session_state` | Khởi tạo state bằng `setdefault` nên không mất tiến trình khi rerun; `knowledge_version` dùng làm key editor để reset text area sau mỗi lần phân tích; vô hiệu hoá report/validation khi knowledge thay đổi | Không xoá được danh sách file khi người dùng gỡ hết file khỏi uploader (`if uploaded:` bỏ qua list rỗng, [app.py:111](../app.py#L111)); khi `PipelineError`, knowledge cũ vẫn hiển thị cạnh thông báo lỗi ([app.py:171-178](../app.py#L171-L178)) |
| `config.py` | Đọc `.env` thành `Settings` (frozen dataclass) | Bất biến, có giá trị mặc định, parse lỗi thì fallback an toàn, không log secret | Giá trị không hợp lệ (ví dụ `CHUNK_SIZE=abc`) bị bỏ qua **im lặng**; không kiểm tra miền giá trị (số âm); `max_chunks` không được dùng |
| `core/file_loader.py` | Đọc file/ZIP → bản ghi chuẩn hoá | Hợp đồng ổn định: luôn trả `dict` có `status`, không ném exception ra ngoài; bảo mật archive tốt (xem 2.2) | PDF không có text layer bị `skipped` mà không gợi ý OCR; `_looks_binary` dựa trên ngưỡng 2% ký tự điều khiển — đủ dùng cho MVP |
| `core/preprocessor.py` | Chuẩn hoá Unicode/whitespace | Có chủ đích **giữ nguyên thụt lề và khoảng trắng nội dòng** — quan trọng với Python/YAML | Không loại bỏ nội dung nhiễu (lockfile, minified JS, file sinh tự động) trước khi tốn token |
| `core/chunker.py` | Chia chunk theo ký tự | Có `ChunkingStrategy` Protocol — điểm mở rộng rõ ràng cho AST/Tree-sitter; kiểm tra bất biến `overlap < size` | Cắt theo ký tự có thể chia đôi hàm/lớp; không có trần số chunk |
| `core/analyzer.py` | MAP: phân tích từng chunk | Filename/chunk_index **lấy từ pipeline, không tin LLM**; chuẩn hoá alias evidence; cache có fallback khi bản cache hỏng | Chạy tuần tự; khoá cache không có namespace model/prompt |
| `core/knowledge_builder.py` | REDUCE: tổng hợp `ProjectKnowledge` | `_apply_evidence_boundary` giới hạn technologies/features/modules trong tập đã quan sát ở bước MAP — cơ chế chống hallucination có hiệu lực thật | Một prompt duy nhất chứa mọi analyses; `architecture.description` không bị ràng buộc evidence; `_merge_configurations` ghi đè khoá trùng giữa các file |
| `core/report_generator.py` | Sinh Markdown 5 chương | Prompt chỉ nhận knowledge JSON; tự chèn chương thiếu thay vì thất bại | Kiểm tra heading bằng `in` (chuỗi con), không kiểm tra thứ tự chương |
| `core/validator.py` | Kiểm tra + revise có giới hạn | Vòng lặp bị chặn cứng; **revision guard** từ chối bản sửa làm phát sinh issue mới | Nhiều heuristic có tỷ lệ dương tính giả cao (xem 3.2) |
| `core/exporter.py` | Tạo tên file an toàn, encode UTF-8 | Slug ASCII, xử lý riêng chữ "Đ/đ" | — |
| `llm/llm_client.py` | Gọi API OpenAI-compatible | Retry có backoff mũ giới hạn (tối đa 8s), `sleep_fn` có thể inject để test; tắt retry nội bộ của SDK (`max_retries=0`) để tránh retry chồng | Retry cả lỗi không thể phục hồi (401, 400); không dùng `response_format` JSON mode; lỗi parse JSON không được retry |
| `models/project_schema.py` | Schema trung tâm | `extra="forbid"` + `str_strip_whitespace`; `Literal` cho confidence/severity | `ChunkAnalysis` dùng `extra="ignore"` (hợp lý); không giới hạn độ dài list/string |
| `utils/cache.py` | Cache JSON theo SHA-256 | Best-effort: lỗi I/O không làm hỏng pipeline | Không có eviction/TTL, không ghi nguyên tử (atomic write) |
| `utils/json_utils.py` | Trích JSON từ phản hồi LLM | Dùng `JSONDecoder.raw_decode` thay vì regex — xử lý đúng chuỗi lồng nhau | Lấy JSON hợp lệ **đầu tiên**, có thể là ví dụ minh hoạ thay vì kết quả |

**Về hỗ trợ đa nhà cung cấp LLM:** Mã nguồn chỉ có một client dựa trên thư viện `openai` với `base_url` cấu hình được. Gemini API chỉ dùng được thông qua endpoint OpenAI-compatible của Google, không có tích hợp Gemini SDK gốc. Tên biến `OPENAI_API_KEY`/`OPENAI_MODEL` cũng phản ánh điều này.

### 2.2. Cơ chế xử lý dữ liệu lớn & Bảo mật

#### 2.2.1. Các ngưỡng an toàn

| Tham số | Mặc định | Được áp dụng tại | Trạng thái |
|---------|---------:|------------------|------------|
| `MAX_FILE_SIZE` | 10 MiB | `_load_raw_file`, từng entry ZIP | ✅ Áp dụng |
| `MAX_ARCHIVE_SIZE` | 50 MiB | `_load_zip_file` (kích thước file nén) | ✅ Áp dụng |
| `MAX_ARCHIVE_EXTRACTED_SIZE` | 100 MiB | Cộng dồn byte **thực sự giải nén** | ✅ Áp dụng |
| `MAX_ARCHIVE_MEMBERS` | 1 000 | Số entry trong ZIP | ✅ Áp dụng |
| `MAX_TOTAL_UPLOAD_SIZE` | 100 MiB | `analyze_project` | ✅ Áp dụng |
| `MAX_CHUNKS` | 500 | — | ❌ **Khai báo nhưng không dùng** |
| `MAX_REVISION` | 1 | `validate_and_revise` | ✅ Áp dụng |
| `REQUEST_TIMEOUT` / `MAX_RETRIES` | 60s / 2 | `LLMClient` | ✅ Áp dụng |

**Hệ quả của việc thiếu `MAX_CHUNKS`:** Với giới hạn 100 MiB văn bản sau giải nén và bước nhảy 5 500 ký tự, một upload hợp lệ có thể sinh ra khoảng **19 000 chunk**, tương ứng 19 000 lời gọi LLM tuần tự trong một request Streamlit. Đây là vấn đề cả về chi phí API lẫn khả năng từ chối dịch vụ (DoS) do chính người dùng gây ra.

```python
# core/pipeline.py:98 — đề xuất bổ sung
chunks = chunk_documents(loadable_files)
if len(chunks) > settings.max_chunks:
    raise PipelineError(
        f"Dự án tạo ra {len(chunks)} chunk, vượt giới hạn {settings.max_chunks}.",
        file_results=file_results,
    )
```

#### 2.2.2. Bảo mật khi giải nén ZIP

Hàm `_load_zip_file` triển khai đầy đủ các biện pháp chống tấn công archive phổ biến:

| Mối đe doạ | Biện pháp | Đánh giá |
|------------|-----------|----------|
| Path traversal (Zip Slip) | `_safe_archive_member_name` từ chối `..`, đường dẫn tuyệt đối, ký tự ổ đĩa Windows, chuẩn hoá `\` → `/` | Tốt. Ngoài ra hệ thống **không ghi file ra đĩa** — entry được đọc thẳng vào bộ nhớ — nên Zip Slip về bản chất không thể khai thác |
| Zip bomb | Đọc tối đa `min(size_limit, remaining) + 1` byte rồi so sánh, **không tin `member.file_size` trong header** | Tốt. Đây là cách làm đúng, vì header ZIP có thể bị giả mạo |
| Symlink | Kiểm tra `S_ISLNK` trên `external_attr` | Tốt |
| Entry mã hoá | Kiểm tra `flag_bits & 0x1` | Tốt |
| ZIP lồng nhau | `.zip` không thuộc `SUPPORTED_EXTENSIONS` → bị bỏ qua | Tốt (tránh bomb đệ quy) |
| Tên trùng | `seen_names` | Tốt |

```python
# core/file_loader.py:404-430 — bounded read, không tin header
with archive.open(member, "r") as entry:
    raw_entry = entry.read(min(size_limit, remaining) + 1)
...
if len(raw_entry) > size_limit or len(raw_entry) > remaining:
    extracted_size = max_archive_extracted_size   # khoá các entry còn lại
```

**Hạn chế:** Toàn bộ archive (tối đa 50 MiB) và mọi upload được giữ dưới dạng `bytes` trong `st.session_state` của mỗi phiên. Với nhiều người dùng đồng thời, mức tiêu thụ RAM tăng tuyến tính theo số phiên.

#### 2.2.3. Cơ chế cache SHA-256

- **Ưu điểm:** Cache theo nội dung (content-addressed) nên tái sử dụng được giữa các file trùng nội dung và giữa các lần "Re-analyze"; `_rebind_cached_payload` gán lại `filename`/`source_file` để evidence không trỏ sai file. Tên file cache là hex digest nên không có rủi ro path injection.
- **Hạn chế:**
  1. **Khoá không có namespace:** Khoá chỉ là `sha256(chunk_content)`. Khi thay đổi `OPENAI_MODEL`, nội dung `prompts/analyze_chunk.txt` hoặc schema, kết quả cũ vẫn được trả về. Khoá nên là `sha256(model ‖ prompt_version ‖ content)`.
  2. **Không có eviction/TTL:** Thư mục `data/cache/` tăng không giới hạn.
  3. **Ghi không nguyên tử:** `write_text` trực tiếp; nếu tiến trình bị dừng giữa chừng, file JSON hỏng sẽ được phát hiện và bỏ qua khi đọc (có xử lý), nhưng nên ghi ra file tạm rồi `os.replace`.

#### 2.2.4. Prompt injection

Nội dung chunk được chèn thô giữa các marker `--- BEGIN CHUNK ---` / `--- END CHUNK ---`. Một file nguồn chứa chuỗi `--- END CHUNK ---` kèm chỉ thị có thể ảnh hưởng đến đầu ra của bước MAP. Rủi ro được **giảm nhẹ một phần** nhờ: filename/chunk_index bị ghi đè bởi pipeline, đầu ra phải qua Pydantic, và bước review thủ công. Đây là rủi ro thấp trong ngữ cảnh người dùng tự phân tích mã nguồn của chính mình.

---

## CHƯƠNG 3: KIỂM ĐỊNH CHẤT LƯỢNG VÀ CƠ CHẾ VALIDATION

### 3.1. Cơ chế Pydantic Validation & Knowledge Building

#### 3.1.1. Chiến lược "chuẩn hoá trước – kiểm tra sau"

Hệ thống không đưa thẳng phản hồi LLM vào Pydantic mà qua một tầng chuẩn hoá có chủ đích (`_prepare_payload`, `_prepare_knowledge_payload`):

| Biến thể từ LLM | Cách xử lý |
|-----------------|-----------|
| Trường list trả về dạng chuỗi đơn | `_as_string_list` bọc thành list |
| Evidence dùng khoá `description`/`source`/`line` | Ánh xạ về `claim`/`source_file`/`source_location` |
| `confidence` ngoài tập cho phép | Đưa về `"medium"` |
| Evidence dạng chuỗi trần | Chuyển thành `Evidence` với `confidence="low"` |
| `purpose` rỗng | Gán `"Không đủ dữ liệu để xác định."` |
| `source_file` không thuộc tập file đã phân tích | Thay bằng file mặc định (bước reduce) |
| Validator trả `severity="critical"` | Ánh xạ thành `"high"` |

Đây là thiết kế phù hợp với bản chất xác suất của LLM: chấp nhận sai lệch về **hình thức** nhưng giữ chặt **ngữ nghĩa** qua schema. Việc đặt `extra="forbid"` cho `ProjectKnowledge` đảm bảo JSON do người dùng sửa ở Step 4 không thể chứa trường lạ.

#### 3.1.2. Evidence boundary — cơ chế chống hallucination ở bước reduce

`_apply_evidence_boundary` ([core/knowledge_builder.py:159-224](../core/knowledge_builder.py#L159-L224)) là thành phần có giá trị kỹ thuật cao nhất của dự án:

```python
technologies = _select_known(candidate.technologies, known_technologies)  # giao với MAP
features     = _select_known(candidate.features, known_features)
modules      = _merge_modules(analyses, known_modules)   # bỏ hẳn modules từ LLM reduce
data_flow / inputs / outputs / limitations / evidence  # lấy trực tiếp từ analyses
```

Bước reduce không thể **thêm** công nghệ, feature hoặc module nào chưa xuất hiện ở bước MAP. Tuy nhiên, phân tích sâu hơn cho thấy:

1. **Phần lớn đầu ra của lời gọi LLM reduce bị loại bỏ.** Chỉ `architecture.description`, thứ tự technologies/features và `architecture.components` (sau khi lọc) được giữ lại. Lời gọi này có chi phí cao (prompt chứa toàn bộ analyses) nhưng đóng góp thông tin hạn chế.
2. **`architecture.description` không bị ràng buộc** — đây là đoạn văn tự do duy nhất từ LLM đi thẳng vào knowledge, và cũng là nơi dễ hallucination nhất.
3. **Mô tả module** được lấy từ `purpose` của chunk **đầu tiên** nhắc tới module đó — không nhất thiết mô tả đúng module.
4. **Gán evidence cho module** dựa trên so khớp chuỗi con (`module_name in evidence.claim`) — dễ bỏ sót hoặc gán nhầm với tên module ngắn.
5. **`_merge_configurations`** dùng `dict.update` → khoá trùng tên giữa các file (ví dụ `PORT`, `DEBUG`) bị ghi đè theo thứ tự xử lý, mất thông tin nguồn gốc.
6. **Knowledge do người dùng sửa không được áp lại evidence boundary.** Đây là lựa chọn thiết kế hợp lý (người dùng là nguồn tin cậy), nhưng cần ghi rõ trong tài liệu rằng validator sẽ coi mọi thứ trong knowledge đã sửa là "đã biết".

### 3.2. Cơ chế Validator & Bounded Revision

#### 3.2.1. Kiến trúc hai tầng

```
validate(report) = dedupe( validate_code(report)  ∪  validate_llm(report) )

validate_code:  _check_structure       — đủ 5 heading chương
                _check_technologies    — regex sau từ khoá "sử dụng/dùng/library/..."
                _check_features        — (a) feature trong knowledge có evidence?
                                         (b) feature nêu trong report có trong knowledge?
                _check_modules         — regex "module X" / "X module"
                _check_unsupported_metrics — số + ngữ cảnh định lượng, trừ dòng có "[source:"
```

**Vòng lặp revision** ([core/validator.py:184-215](../core/validator.py#L184-L215)) có ba tính chất đúng đắn:
- **Kết thúc chắc chắn:** `while has_high_issue and revision_count < limit`, `limit = max(0, MAX_REVISION)`.
- **Có guard:** `_revision_is_allowed` từ chối bản sửa nếu thay đổi chữ ký chương hoặc phát sinh issue `high` mới so với bản gốc → tránh hiện tượng "sửa một lỗi, sinh hai lỗi".
- **Fail-safe:** Khi guard từ chối, giữ nguyên bản cũ và gắn thêm issue giải thích.

#### 3.2.2. Kết quả thực nghiệm trên các heuristic

Các hàm kiểm tra được chạy trực tiếp với `ProjectKnowledge(technologies=["Python","Streamlit","Pydantic"], features=["Upload file ZIP","Sinh báo cáo Markdown"])` và một evidence `"app.py dùng st.file_uploader để nhận file zip"`:

| Đầu vào | Hàm | Kết quả thực tế | Kỳ vọng | Đánh giá |
|---------|-----|-----------------|---------|----------|
| Report **rỗng** `""` | `_check_features` | 2 issue `high`: `Upload file ZIP`, `Sinh báo cáo Markdown` | 0 issue | ❌ Lỗi logic |
| "Hệ thống sử dụng Python và Streamlit để xây dựng giao diện người dùng." | `_check_technologies` | 0 issue | 0 issue | ✅ |
| "Người dùng có thể dùng giao diện web để tải lên tài liệu." | `_check_technologies` | 2 issue `high` (`"có thể dùng giao diện web..."`) | 0 issue | ❌ Dương tính giả |
| "Module xử lý dữ liệu đọc file đầu vào." | `_check_modules` | 1 issue `high`: `xử lý dữ liệu` | Tuỳ ngữ cảnh | ⚠️ Nhạy với văn phong |
| "Độ chính xác đạt 99% [source: fake.py]" | `_check_unsupported_metrics` | 0 issue | ≥ 1 issue | ❌ Có thể vượt qua |
| "Pipeline gồm 7 bước xử lý tuần tự." | `_check_unsupported_metrics` | 0 issue | Tuỳ ngữ cảnh | ⚠️ |

**Phân tích nguyên nhân:**

1. **Kiểm tra feature (a) không phụ thuộc report.** Đoạn mã kiểm tra xem mỗi `feature` có xuất hiện nguyên văn (chuỗi con) trong các `evidence.claim` hay không. Vì claim là văn bản tự do, điều kiện này gần như luôn sai. Quan trọng hơn, đây là thuộc tính của **knowledge**, nên **không bản revise nào có thể khắc phục** — mỗi lần validate luôn có issue `high`, vòng revision luôn chạy tới giới hạn, và kết quả cuối luôn là `valid=False`.

   ```python
   # core/validator.py:278-286
   for feature in knowledge.features:
       if feature.casefold() not in " ".join(evidence_claims):   # không dùng `report`
           issues.append(ValidationIssue(..., severity="high"))
   ```
   Đề xuất: chuyển kiểm tra này sang bước knowledge building (cảnh báo khi review), hoặc hạ xuống `severity="low"` và loại khỏi điều kiện kích hoạt revision.

2. **Regex công nghệ khớp từ "dùng" trong "người dùng".** Cụm "người dùng" xuất hiện với tần suất rất cao trong báo cáo tiếng Việt, khiến phần còn lại của câu bị tách thành "ứng viên công nghệ". Nên loại trừ bằng negative lookbehind `(?<!người )dùng` hoặc chuyển sang so khớp ngược: chỉ cảnh báo khi một **tên riêng dạng công nghệ** (CamelCase, có dấu chấm, nằm trong từ điển công nghệ) không thuộc knowledge.

3. **Cơ chế miễn trừ `[source:` quá dễ dãi.** Bất kỳ dòng nào chứa `[source:` đều được bỏ qua kiểm tra số liệu, kể cả khi file nguồn không tồn tại. Cần đối chiếu tên file trong thẻ với `knowledge.source_files`.

4. **LLM validator lỗi thì toàn bộ validation thất bại.** `validate_llm` ném `ValidatorError` thay vì trả về kết quả deterministic kèm cảnh báo. Nên có chế độ suy giảm (graceful degradation).

#### 3.2.3. Đánh giá chung về tầng validation

Kiến trúc (deterministic + LLM, revision có giới hạn, guard chống thoái hoá) là **đúng hướng và vượt mức thông thường của một MVP**. Tuy nhiên, tỷ lệ dương tính giả cao làm giảm giá trị tín hiệu: khi gần như mọi báo cáo đều bị đánh `Invalid`, người dùng sẽ có xu hướng bỏ qua kết quả validation. Đồng thời, mỗi issue giả kích hoạt một lời gọi revise tốn chi phí và có thể làm báo cáo bị sửa không cần thiết.

#### 3.2.4. Độ phủ kiểm thử

| Nhóm | File test | Nội dung chính |
|------|-----------|----------------|
| Nạp file | `test_file_loader.py`, `test_zip_pipeline.py` | Định dạng, file rỗng, binary, ZIP traversal, giới hạn |
| Tiền xử lý & chunk | `test_preprocessor.py`, `test_chunker.py` | Chuẩn hoá, offset, overlap |
| LLM & JSON | `test_llm_client.py`, `test_json_utils.py` | Mock client, trích JSON |
| Phân tích & tri thức | `test_analyzer.py`, `test_knowledge_builder.py`, `test_project_schema.py` | Schema, evidence boundary |
| Sinh & kiểm tra | `test_report_generator.py`, `test_validator.py` | Heading, phát hiện issue |
| Tích hợp | `test_end_to_end.py`, `test_cache_exporter.py` | Pipeline đầy đủ với mock |

**Kết quả:** `35 passed in 3.46s`. Các khoảng trống: chưa có test cho dương tính giả của validator (các trường hợp ở mục 3.2.2), chưa có test cho `MAX_CHUNKS`, chưa có test cho logic retry của `LLMClient` khi gặp lỗi không thể phục hồi, chưa có test UI (`streamlit.testing.v1.AppTest`).

---

## CHƯƠNG 4: ĐÁNH GIÁ ƯU ĐIỂM VÀ HẠN CHẾ (PROS & CONS)

### 4.1. Ưu điểm nổi bật

1. **Knowledge-driven generation có kiểm chứng.** Report generator không bao giờ thấy raw source code; mọi nhận định phải đi qua `ProjectKnowledge` đã bị ràng buộc bởi evidence boundary và đã được con người duyệt. Đây là cơ chế kiểm soát hallucination có cơ sở kỹ thuật rõ ràng, không chỉ dựa vào chỉ dẫn trong prompt.
2. **Human-in-the-loop đúng vị trí.** Điểm review nằm giữa hai pha — nơi chi phí sửa sai thấp nhất và tác động lớn nhất. Dữ liệu sửa đổi được kiểm tra bằng Pydantic trước khi lưu, và report/validation phía sau được vô hiệu hoá tự động.
3. **Xử lý lỗi theo nguyên tắc cô lập.** File lỗi trở thành bản ghi `status="error"`, chunk lỗi được gom vào `AnalysisRun.errors`; pipeline chỉ dừng khi không còn dữ liệu có ý nghĩa. Giao diện hiển thị trạng thái từng file và lỗi từng chunk, giúp người dùng chẩn đoán.
4. **Bảo mật tầng nạp file tốt.** Bounded read không tin header, chặn symlink/mã hoá/traversal, không ghi đĩa.
5. **Khả năng kiểm thử cao.** Dependency injection nhất quán (`llm_client`, `cache`, `settings`, `prompt_path`, `sleep_fn`); `MockLLMClient` dạng hàng đợi cho phép test toàn bộ pipeline offline và deterministic.
6. **Prompt tách khỏi mã nguồn** (`prompts/*.txt`), thay thế placeholder bằng `str.replace` thay vì `str.format` — tránh lỗi khi source code chứa dấu `{}`.
7. **Logging an toàn:** Chỉ ghi metadata (tên file, kích thước, số lượng), không ghi nội dung hay API key; có rotating handler.
8. **Giao diện thực tế:** Luồng 6 bước tuyến tính, thanh tiến trình theo từng chunk, xem trước Markdown/Raw/Validation, tải file trực tiếp.

### 4.2. Hạn chế còn tồn tại

#### 4.2.1. Hạn chế kiến trúc (giới hạn MVP)

| Hạn chế | Hệ quả | Mức độ |
|---------|--------|--------|
| **Xử lý đồng bộ trong request Streamlit**, không có job queue | Phân tích N chunk chặn toàn bộ phiên; đóng tab hoặc rerun làm mất tiến trình đang chạy (dù cache giảm thiểu phần đã xong); không thể mở rộng ngang | Cao |
| **Phân tích chunk tuần tự** | Thời gian tăng tuyến tính: 200 chunk × ~5s ≈ 17 phút | Cao |
| **Reduce một lần (single-shot)** | Dự án lớn có thể vượt context window ở bước knowledge building; không có reduce phân cấp | Trung bình |
| **Chỉ trích xuất text layer của PDF** | PDF scan/ảnh bị đánh dấu `skipped` ("Không tìm thấy nội dung văn bản"); chưa có OCR | Trung bình |
| **Chunk theo ký tự** | Có thể cắt ngang hàm/lớp; LLM thấy ngữ cảnh không trọn vẹn → phân tích kém chính xác | Trung bình |
| **Không có ngữ cảnh liên file** | Mỗi chunk được phân tích cô lập; quan hệ import/gọi hàm giữa các module không được mô hình hoá | Trung bình |
| **Trạng thái nằm trong `st.session_state`** | Không lưu bền; restart server mất dữ liệu; RAM tăng theo số phiên × kích thước upload | Trung bình |
| **Chỉ hỗ trợ API OpenAI-compatible** | Gemini/Claude chỉ dùng qua endpoint tương thích; không tận dụng structured output gốc | Thấp |
| **Chỉ xuất Markdown** | `exporter.py` chưa hỗ trợ DOCX/PDF dù `python-docx` đã có trong dependencies | Thấp |

#### 4.2.2. Lỗi và khiếm khuyết cụ thể trong mã nguồn

| # | Vấn đề | Vị trí |
|---|--------|--------|
| 1 | `MAX_CHUNKS` không được áp dụng | [config.py:33](../config.py#L33) |
| 2 | Kiểm tra feature đánh giá knowledge thay vì report → issue `high` không thể sửa | [core/validator.py:278-286](../core/validator.py#L278-L286) |
| 3 | Dương tính giả với "người dùng" | [core/validator.py:22-26](../core/validator.py#L22-L26) |
| 4 | Miễn trừ `[source:` không kiểm tra file tồn tại | [core/validator.py:338](../core/validator.py#L338) |
| 5 | LLM validator lỗi → toàn bộ validation lỗi | [core/validator.py:126-132](../core/validator.py#L126-L132) |
| 6 | Retry mọi exception, kể cả 401/400 | [llm/llm_client.py:98](../llm/llm_client.py#L98) |
| 7 | Lỗi parse JSON không được retry | [llm/llm_client.py:122-127](../llm/llm_client.py#L122-L127) |
| 8 | Khoá cache không gồm model/prompt version | [core/analyzer.py:79-80](../core/analyzer.py#L79-L80) |
| 9 | `architecture.description` không bị ràng buộc evidence | [core/knowledge_builder.py:203-207](../core/knowledge_builder.py#L203-L207) |
| 10 | Cấu hình trùng khoá bị ghi đè giữa các file | [core/knowledge_builder.py:266-270](../core/knowledge_builder.py#L266-L270) |
| 11 | Không xoá được danh sách file đã upload khi gỡ hết | [app.py:111](../app.py#L111) |
| 12 | Knowledge cũ vẫn hiển thị khi lần phân tích mới thất bại | [app.py:171-178](../app.py#L171-L178) |
| 13 | Hàm tiện ích trùng lặp (`_as_string_list`, `_unique_strings`, hằng `NOT_ENOUGH_DATA`) giữa `analyzer`, `knowledge_builder`, `report_generator`, `validator`; `_unique_strings` trả `set` ở một nơi và `list` ở nơi khác | Nhiều file |

---

## CHƯƠNG 5: KẾT LUẬN VÀ KHUYẾN NGHỊ PHÁT TRIỂN

### 5.1. Kết luận chung

**Bảng chấm điểm theo tiêu chí:**

| Tiêu chí | Điểm (/10) | Nhận xét |
|----------|:----------:|----------|
| Kiến trúc & thiết kế pipeline | 8.5 | Map–reduce + tri thức trung gian + human review là thiết kế đúng cho bài toán; điểm mở rộng (`ChunkingStrategy`, DI) rõ ràng |
| Độ tin cậy & xử lý lỗi | 7.5 | Cô lập lỗi tốt, bảo mật archive tốt; trừ điểm do `MAX_CHUNKS` không hiệu lực và retry chưa phân loại lỗi |
| Chất lượng mã nguồn | 8.0 | Type hints đầy đủ, docstring rõ, module nhỏ, tách prompt; còn trùng lặp tiện ích |
| Cơ chế validation | 6.0 | Kiến trúc tốt nhưng heuristic có tỷ lệ dương tính giả cao làm giảm giá trị thực tế |
| Kiểm thử | 7.5 | 35 test pass, chạy offline; thiếu test cho các trường hợp biên của validator và UI |
| Khả năng mở rộng (scalability) | 5.0 | Đồng bộ, tuần tự, state trong bộ nhớ — phù hợp demo một người dùng, chưa phù hợp triển khai nhiều người dùng |
| **Tổng hợp** | **7.1** | |

**Mức độ sẵn sàng:**
- **Demo / bảo vệ đồ án / sử dụng cá nhân:** Sẵn sàng. Pipeline hoạt động end-to-end, có giao diện hoàn chỉnh và bộ test đạt.
- **Triển khai cho nhóm nhỏ (nội bộ lớp/khoa):** Cần khắc phục tối thiểu các mục 1–3 và 6 ở bảng 4.2.2.
- **Triển khai production nhiều người dùng:** Chưa sẵn sàng; cần job queue, lưu trữ bền vững, xác thực người dùng và giới hạn tần suất.

### 5.2. Khuyến nghị hướng phát triển

Các khuyến nghị được sắp xếp theo lộ trình, từ sửa lỗi chi phí thấp đến nâng cấp kiến trúc.

#### Giai đoạn 1 — Sửa lỗi và gia cố (ngắn hạn)

| Hạng mục | Thay đổi đề xuất |
|----------|------------------|
| Áp dụng `MAX_CHUNKS` | Kiểm tra sau `chunk_documents` trong `pipeline.py`, báo lỗi rõ ràng kèm số chunk thực tế |
| Sửa `_check_features` | Bỏ kiểm tra (a) khỏi validator report; chuyển thành cảnh báo ở Step 4 |
| Giảm dương tính giả | Loại trừ "người dùng"; chuyển sang chiến lược "trích tên riêng → đối chiếu" thay vì "trích mọi thứ sau từ khoá" |
| Kiểm tra thẻ `[source:]` | Đối chiếu tên file với `knowledge.source_files` |
| Phân loại lỗi retry | Chỉ retry `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`; retry cả khi parse JSON thất bại |
| Namespace cache | `key = sha256(f"{model}\n{prompt_sha}\n{content}")` |
| JSON mode | Truyền `response_format={"type": "json_object"}` khi endpoint hỗ trợ |
| Hợp nhất tiện ích | Tạo `utils/text_utils.py` chứa `as_string_list`, `unique_strings`, `NOT_ENOUGH_DATA` |

#### Giai đoạn 2 — Nâng cao chất lượng phân tích (trung hạn)

**a) Tree-sitter AST chunking.** Interface `ChunkingStrategy` đã sẵn sàng; chỉ cần thêm một lớp mới mà không sửa pipeline:

```python
class TreeSitterChunker:
    """Chia theo node cấp cao (function/class), gộp node nhỏ tới chunk_size,
    fallback về CharacterChunker cho ngôn ngữ không hỗ trợ hoặc node quá lớn."""

    def chunk(self, filename: str, content: str) -> list[dict[str, Any]]:
        tree = self._parser_for(filename).parse(content.encode())
        ...  # giữ start_char/end_char và thêm "symbol": "ClassName.method"

chunks = chunk_documents(loadable_files, strategy=TreeSitterChunker())
```

Lợi ích: chunk mang ngữ nghĩa trọn vẹn; trường `symbol` cung cấp `source_location` chính xác cho evidence; có thể trích xuất đồ thị import để mô tả quan hệ giữa các module.

**b) Reduce phân cấp.** Gom analyses theo thư mục/module → reduce cấp 1 cho từng nhóm → reduce cấp 2 cho toàn dự án. Giải quyết giới hạn context window và cho phép song song hoá.

**c) Vector DB / RAG cho bước sinh báo cáo.** Lưu embedding của chunk (ví dụ Chroma, Qdrant, pgvector). Khi sinh từng chương, truy xuất evidence liên quan nhất thay vì đưa toàn bộ knowledge vào prompt. Lưu ý: để giữ tính chất chống hallucination hiện tại, RAG chỉ nên dùng để **truy xuất evidence** đưa vào knowledge, không cho generator truy cập raw code trực tiếp.

**d) Sinh báo cáo theo từng chương** (5 lời gọi nhỏ thay vì một lời gọi lớn), cho phép validate và revise ở mức chương, giảm phạm vi ảnh hưởng của mỗi lần revise.

**e) OCR cho PDF scan.** Phát hiện trang không có text layer (`page.get_text()` rỗng) và chuyển sang OCR (PyMuPDF + Tesseract qua `page.get_textpage_ocr()`, hoặc mô hình vision của LLM).

#### Giai đoạn 3 — Kiến trúc triển khai (dài hạn)

```
┌───────────┐  POST /jobs   ┌───────────┐  enqueue  ┌──────────────┐
│ Streamlit │──────────────►│  FastAPI  │──────────►│ Redis / RQ   │
│  (UI)     │◄──poll/SSE────│  (API)    │           │ hoặc Celery  │
└───────────┘               └─────┬─────┘           └──────┬───────┘
                                  │                        ▼
                           ┌──────▼──────┐        ┌────────────────┐
                           │ PostgreSQL  │◄───────│ Worker pool    │
                           │ job, state, │        │ asyncio + sem. │
                           │ knowledge   │        │ (song song     │
                           └─────────────┘        │  hoá MAP)      │
                                                  └────────────────┘
```

| Hạng mục | Mô tả |
|----------|-------|
| Job queue bất đồng bộ | Tách phân tích khỏi request UI; người dùng có thể đóng tab và quay lại |
| Song song hoá MAP | `AsyncOpenAI` + `asyncio.Semaphore(k)` để giới hạn đồng thời theo rate limit nhà cung cấp; kỳ vọng giảm thời gian xấp xỉ k lần |
| Lưu trữ bền vững | Knowledge, report, lịch sử revision lưu trong CSDL thay vì `session_state` |
| Đa nhà cung cấp | Tách `LLMClient` thành Protocol với các adapter riêng (OpenAI, Gemini SDK, Anthropic) và structured output gốc của từng nhà cung cấp |
| Quan sát hệ thống | Ghi số token, chi phí, độ trễ từng lời gọi; hiển thị chi phí ước tính trước khi chạy |
| Xuất đa định dạng | DOCX (đã có `python-docx`), PDF theo template của trường |

---

## PHỤ LỤC A: CÁC LỖI ĐÃ KHẮC PHỤC

### A.1. Tổng quan

Sau đánh giá, 11 trong 13 khiếm khuyết liệt kê ở mục 4.2.2 đã được khắc phục. Mỗi bản sửa đi kèm ít nhất một test hồi quy. Bộ kiểm thử tăng từ **35 lên 49 test, toàn bộ đạt**. Giao diện được kiểm tra khói (smoke test) bằng `streamlit.testing.v1.AppTest`: khởi động không phát sinh exception và nhánh xử lý lỗi phân tích hoạt động đúng.

### A.2. Bảng đối chiếu khắc phục

| # (4.2.2) | Khiếm khuyết | Trạng thái | Cách khắc phục | Test hồi quy |
|:---------:|--------------|:----------:|----------------|--------------|
| 1 | `MAX_CHUNKS` không được áp dụng | ✅ | `analyze_project` ném `PipelineError` khi số chunk vượt `MAX_CHUNKS`, **trước** mọi lời gọi LLM | `test_pipeline_limits.py` |
| 2 | Kiểm tra feature đánh giá knowledge thay vì report | ✅ | Loại khỏi validator report; tách thành hàm công khai `find_features_without_evidence()`, hiển thị dạng cảnh báo tại Step 4 | `test_empty_report_has_no_feature_issue_from_knowledge_alone` |
| 3 | Dương tính giả với "người dùng" | ✅ | Negative lookbehind cho "người dùng"/"người sử dụng"; chỉ coi là tên công nghệ/module khi cụm từ có dạng định danh (xem A.3); kiểm tra feature chỉ áp dụng cho dạng liệt kê `Tính năng: A, B` | `test_vietnamese_prose_does_not_create_false_positives`, `test_unknown_technology_after_prose_is_still_detected` |
| 4 | Miễn trừ `[source:` không kiểm tra file tồn tại | ✅ | Thêm `_check_source_references`: mọi thẻ `[source: X]` được đối chiếu với `knowledge.source_files` (chấp nhận khớp đầy đủ hoặc hậu tố đường dẫn, ví dụ `auth.py` ↔ `demo.zip::src/auth.py`); kiểm tra số liệu chỉ được miễn khi **mọi** thẻ trên dòng đều hợp lệ | `test_fake_source_tag_does_not_bypass_metric_check`, `test_metric_with_known_source_tag_is_accepted` |
| 5 | LLM validator lỗi → toàn bộ validation lỗi | ✅ | `validate()` trả kết quả deterministic kèm issue `medium`; `validate_and_revise()` giữ nguyên báo cáo khi bước revise thất bại | `test_llm_validator_failure_falls_back_to_deterministic_result`, `test_revision_failure_keeps_current_report` |
| 6 | Retry mọi exception, kể cả 401/400 | ✅ | `_is_retryable()`: không retry `APIStatusError` 4xx, trừ 408/409/429 | `test_authentication_error_is_not_retried`, `test_rate_limit_error_is_retried` |
| 7 | Lỗi parse JSON không được retry | ✅ | `generate_json()` yêu cầu lại tối đa `MAX_RETRIES` lần khi phản hồi không chứa JSON hợp lệ | `test_invalid_json_response_is_requested_again`, `test_invalid_json_is_raised_after_retries_are_exhausted` |
| 8 | Khoá cache không gồm model/prompt | ✅ | Khoá = `sha256(namespace : content_hash)`, với `namespace = sha256(schema_version ‖ model ‖ sha256(prompt))` | `test_cache_is_not_reused_after_prompt_changes` |
| 9 | `architecture.description` không bị ràng buộc evidence | ⏸ Chưa xử lý | Vấn đề thiết kế: đoạn mô tả tự do không thể đối chiếu tập hợp như danh sách công nghệ; cần hướng tiếp cận riêng (sinh mô tả từ components đã kiểm chứng) | — |
| 10 | Cấu hình trùng khoá bị ghi đè | ✅ | Giá trị đầu tiên giữ khoá gốc; giá trị khác từ file khác được lưu dưới khoá `"KEY [filename]"` | `test_conflicting_configuration_values_are_kept_per_file` |
| 11 | Không xoá được danh sách file đã upload | ✅ | Đồng bộ `session_state.uploaded_files` với uploader ở mọi lần rerun | Smoke test AppTest |
| 12 | Knowledge cũ hiển thị khi phân tích thất bại | ✅ | Hàm `_clear_analysis_state()` xoá knowledge, report, validation, thống kê của lần chạy trước | Smoke test AppTest |
| 13 | Hàm tiện ích trùng lặp | ✅ | Hợp nhất `NOT_ENOUGH_DATA`, `as_string_list`, `unique_strings` vào `utils/text_utils.py` | Toàn bộ test hiện có |

### A.3. Kết quả thực nghiệm lặp lại trên validator

Các trường hợp ở mục 3.2.2 được chạy lại sau khi sửa:

| Đầu vào | Trước khắc phục | Sau khắc phục |
|---------|-----------------|---------------|
| Report rỗng, knowledge có 2 feature | 2 issue `high` | 0 issue (cảnh báo chuyển sang Step 4) |
| "Người dùng có thể dùng giao diện web để tải lên tài liệu." | 2 issue `high` | 0 issue |
| "Module xử lý dữ liệu đọc file đầu vào." | 1 issue `high` | 0 issue |
| "Hệ thống có chức năng tải file lên." | 1 issue `high` | 0 issue |
| "Hệ thống dùng Rust để xử lý dữ liệu." (Rust không có trong knowledge) | Issue với cụm dài `Rust để xử lý dữ liệu` | 1 issue `high`: `Rust` |
| "Độ chính xác đạt 99% [source: fake.py]" | 0 issue | 2 issue `high` (source không tồn tại + số liệu không có nguồn) |
| "Thời gian xử lý đạt 120 ms [source: auth.py]" (file có thật) | 0 issue | 0 issue |

**Quy tắc nhận diện định danh** dùng cho công nghệ và module: lấy tối đa 3 từ đầu tiên của cụm ứng viên, mỗi từ phải là ký tự ASCII và có chữ hoa, chữ số hoặc một trong các ký tự `_ . + # / -` (ví dụ `Python`, `Node.js`, `C++`, `core/analyzer.py`). Văn xuôi tiếng Việt có nhiều âm tiết ASCII viết thường (`giao`, `web`, `file`) nên bị loại.

**Đánh đổi:** tên công nghệ viết thường hoàn toàn (ví dụ `pytest`, `numpy`) không còn bị bắt bằng regex. Trường hợp này được giao cho LLM validator.

### A.4. Thay đổi hành vi cần lưu ý khi triển khai

- **Cache cũ bị vô hiệu:** Do khoá cache thay đổi, các bản ghi trong `data/cache/` tạo trước khi sửa không còn được sử dụng. Lần chạy đầu tiên sau khi cập nhật sẽ gọi lại LLM cho mọi chunk. Thư mục này có thể xoá an toàn.
- **Dự án lớn có thể bị từ chối:** Upload sinh ra hơn `MAX_CHUNKS` (mặc định 500) chunk sẽ dừng ngay với thông báo hướng dẫn tăng `CHUNK_SIZE`/`MAX_CHUNKS` hoặc loại bớt file.
- **Kết quả validation không còn luôn là `Invalid`:** Do loại bỏ các issue không thể sửa và giảm dương tính giả, vòng revision chỉ được kích hoạt khi có vấn đề thực sự trong báo cáo.

### A.5. Hạng mục còn tồn tại

- Mục 4 trong bảng tóm tắt (reduce một lần) và toàn bộ hạn chế kiến trúc ở mục 4.2.1 (xử lý đồng bộ, chưa có OCR, chunk theo ký tự) vẫn giữ nguyên. Lộ trình khắc phục xem mục 5.2.
- Khi phân tích thất bại, thông báo lỗi hiển thị hai lần (thanh trạng thái và khu vực kết quả). Đây là hành vi có sẵn từ phiên bản gốc, chưa thay đổi.

---

*Báo cáo được lập dựa trên phân tích tĩnh mã nguồn, kết quả chạy `pytest` (35/35 đạt trước khắc phục, 49/49 đạt sau khắc phục) và thực nghiệm trực tiếp trên các hàm validator. Các nhận định về hiệu năng (thời gian xử lý, số chunk tối đa) là ước tính tính toán từ cấu hình mặc định, chưa được đo trên tải thực tế.*
