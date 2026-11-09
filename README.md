# AI Technical Report Generator

AI Technical Report Generator là ứng dụng Streamlit hỗ trợ phân tích source code,
file cấu hình và tài liệu kỹ thuật để tạo báo cáo đồ án Markdown bằng tiếng Việt.

Ứng dụng không gửi toàn bộ source code trong một request duy nhất. Dữ liệu được
đọc, tiền xử lý, chia thành chunk và phân tích theo từng chunk trước khi tổng hợp
thành `Project Knowledge` để người dùng kiểm tra.

## 1. Yêu cầu môi trường

- Windows 10 hoặc Windows 11.
- Python 3.11 trở lên.
- Kết nối mạng nếu sử dụng LLM API bên ngoài.
- Một API key của OpenAI hoặc nhà cung cấp OpenAI-compatible API.

Kiểm tra Python:

```powershell
python --version
```

## 2. Cài đặt

Mở PowerShell tại thư mục project:

```powershell
cd D:\NCKH_SinhBaoCao
```

Tạo virtual environment:

```powershell
python -m venv .venv
```

Kích hoạt virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chỉ mở quyền cho phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Cài dependency:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Cấu hình `.env`

Tạo file `.env` từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Mở `.env` và điền API key:

```env
LLM_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
MAX_ARCHIVE_SIZE=52428800
MAX_ARCHIVE_EXTRACTED_SIZE=104857600
MAX_ARCHIVE_MEMBERS=1000
```

Với nhà cung cấp OpenAI-compatible khác, thay `LLM_BASE_URL` và `OPENAI_MODEL`
theo tài liệu của nhà cung cấp.

Không commit `.env`. File này đã được thêm vào `.gitignore`.

## 4. Chạy ứng dụng

```powershell
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trên terminal, thường là:

```text
http://localhost:8501
```

## 5. Cách sử dụng

1. Nhập Project Title, Subject Name, Technology Stack và Additional Notes.
2. Upload nhiều file source code hoặc tài liệu.
3. Bấm **Analyze Project**.
4. Kiểm tra trạng thái file, số chunk và các metrics.
5. Xem hoặc chỉnh sửa `Project Knowledge JSON`.
6. Bấm **Validate and Save Project Knowledge** nếu đã chỉnh sửa.
7. Bấm **Generate Report**.
8. Xem báo cáo ở tab **Preview** hoặc **Raw Markdown**.
9. Bấm **Validate Report** để kiểm tra claim và revise tự động trong giới hạn cấu hình.
10. Tải báo cáo bằng nút **Download Markdown**.

## 6. Kiến trúc và pipeline

```text
User
  ↓
Streamlit UI
  ↓
File Loader
  ↓
Preprocessor
  ↓
Chunker
  ↓
Chunk Analyzer
  ↓
Project Knowledge Builder
  ↓
Human Review
  ↓
Report Generator
  ↓
Deterministic Validator + LLM Validator
  ↓
Bounded Revision
  ↓
Final Markdown Report
```

Nguyên tắc chính:

```text
SOURCE DATA → FACT EXTRACTION → STRUCTURED KNOWLEDGE → REPORT
```

Report Generator chỉ nhận `ProjectKnowledge`, không nhận raw source làm nguồn
chính.

## 7. Các module chính

| Module | Trách nhiệm |
|---|---|
| `app.py` | Giao diện Streamlit và session state |
| `config.py` | Đọc cấu hình từ environment và `.env` |
| `core/file_loader.py` | Đọc source, PDF, DOCX và metadata file |
| `core/preprocessor.py` | Chuẩn hóa encoding, newline và whitespace |
| `core/chunker.py` | Chia tài liệu thành chunk có metadata |
| `core/analyzer.py` | Gọi LLM phân tích từng chunk |
| `core/knowledge_builder.py` | Chuẩn hóa, deduplicate và reduce analysis |
| `core/pipeline.py` | Điều phối pipeline phân tích |
| `core/report_generator.py` | Sinh báo cáo Markdown 5 chương |
| `core/validator.py` | Kiểm tra report và revise có giới hạn |
| `core/exporter.py` | Tạo tên file và export Markdown |
| `llm/llm_client.py` | OpenAI-compatible client và mock client |
| `models/project_schema.py` | Pydantic schemas trung tâm |
| `utils/cache.py` | Cache theo SHA-256 |
| `utils/json_utils.py` | Extract JSON từ LLM response |
| `utils/logging_config.py` | Cấu hình Python logging |

## 8. File đầu vào hỗ trợ

Source/text:

```text
.py .java .kt .cpp .c .h .hpp .js .ts .jsx .tsx
.json .xml .yaml .yml .md .txt .properties .gradle
.sql .html .css
```

Tài liệu:

```text
.pdf .docx
.zip
```

PDF cần có text layer. PDF chỉ chứa ảnh scan chưa được OCR trong MVP.

File `.zip` được giải nén trong bộ nhớ trước khi phân tích. Hệ thống chỉ giữ các
file có extension được hỗ trợ, chặn path traversal, symbolic link, entry mã hóa,
file vượt `MAX_FILE_SIZE`, số lượng entry vượt `MAX_ARCHIVE_MEMBERS` và tổng dung
lượng sau giải nén vượt `MAX_ARCHIVE_EXTRACTED_SIZE`. Tên nguồn trong evidence có
dạng `project.zip::folder/file.py`.

## 9. Cache và logging

Cache phân tích được lưu tại:

```text
data/cache/
```

Log ứng dụng được ghi tại:

```text
data/outputs/app.log
```

Cache dùng SHA-256 nội dung chunk. Nếu cache hỏng hoặc không thể ghi, hệ thống
bỏ qua cache và phân tích lại.

Log không ghi API key hoặc raw source content.

## 10. Chạy test

Chạy toàn bộ test:

```powershell
pytest -q
```

Test bao gồm:

- File loader.
- Preprocessor.
- Chunker.
- JSON extractor.
- Pydantic schema.
- Chunk analyzer.
- Knowledge builder.
- Report generator.
- Validator và revision guard.
- Cache và exporter.
- End-to-end pipeline bằng `MockLLMClient`, không cần API thật.

Kiểm tra syntax toàn bộ Python:

```powershell
python -m compileall -q app.py config.py core llm models utils tests examples
```

## 11. Giới hạn của MVP

- Chạy đồng bộ, chưa có job queue hoặc xử lý parallel thực sự.
- Chunker hiện tại dựa trên ký tự, chưa dùng AST hoặc Tree-sitter.
- Chưa có authentication và user account.
- Chưa có database server để lưu project và lịch sử report.
- Chưa có Docker, Kubernetes hoặc microservices.
- Chưa có RAG, embedding hoặc vector database.
- Chưa có OCR cho PDF scan.
- Chưa export DOCX/PDF.
- Validator giúp phát hiện và giảm hallucination nhưng không thể bảo đảm tuyệt đối.
- Source code vẫn được gửi tới LLM theo từng chunk nếu dùng API bên ngoài.
- Streamlit session state phù hợp local/demo, chưa phải cơ chế lưu trữ lâu dài.
- Chi phí và thời gian xử lý tăng theo số lượng file, chunk và số vòng revision.

## 12. Hướng phát triển

- AST/Tree-sitter semantic chunking.
- Parallel chunk analysis.
- RAG với embedding và vector database.
- Citation chi tiết tới file và dòng code.
- Mermaid architecture diagram.
- Export DOCX/PDF.
- FastAPI backend và job queue.
- Database lưu project, analysis và lịch sử report.
- OCR cho PDF scan.
- Secret redaction trước khi gửi source tới LLM.
- LLM local cho project có yêu cầu bảo mật cao.
