"""Streamlit UI for the AI Technical Report Generator."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from pydantic import ValidationError

from core.pipeline import PipelineError, analyze_project
from core.exporter import build_report_filename, docx_bytes, markdown_bytes, pdf_bytes
from core.report_generator import ReportGenerationError, generate_report
from core.validator import (
    ValidatorError,
    find_features_without_evidence,
    validate_and_revise_report,
)
from models.project_schema import ProjectKnowledge
from utils.logging_config import configure_logging


SUPPORTED_UPLOAD_TYPES = [
    "py",
    "java",
    "kt",
    "cpp",
    "c",
    "h",
    "hpp",
    "js",
    "ts",
    "jsx",
    "tsx",
    "json",
    "xml",
    "yaml",
    "yml",
    "md",
    "txt",
    "properties",
    "gradle",
    "kts",
    "sql",
    "html",
    "css",
    "pdf",
    "docx",
    "zip",
]
_FILE_STATUS_LABELS = {
    "loaded": "Đã đọc",
    "skipped": "Bỏ qua",
    "error": "Lỗi",
    "expanded": "Đã giải nén",
}
_SEVERITY_LABELS = {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}


def main() -> None:
    """Render the application and preserve all pipeline state across reruns."""
    configure_logging()
    _initialize_session_state()
    st.set_page_config(
        page_title="Sinh báo cáo kỹ thuật bằng AI",
        page_icon="📄",
        layout="wide",
    )

    st.title("Sinh báo cáo kỹ thuật bằng AI")
    st.caption(
        "Phân tích mã nguồn và tài liệu theo quy trình nhiều bước, "
        "có bước người dùng kiểm duyệt."
    )

    _render_project_information()
    _render_file_upload()
    _render_analyze_step()
    _render_analysis_results()
    _render_knowledge_review()
    _render_report_step()
    _render_validation_step()


def _initialize_session_state() -> None:
    """Initialize keys once so Streamlit reruns do not erase user progress."""
    defaults: dict[str, Any] = {
        "uploaded_files": [],
        "chunk_analysis": [],
        "project_knowledge": None,
        "report": None,
        "validation_result": None,
        "file_results": [],
        "analysis_errors": [],
        "analysis_stats": {},
        "analysis_error": None,
        "knowledge_version": 0,
        "revision_count": 0,
        "validation_error": None,
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


def _render_project_information() -> None:
    st.subheader("Bước 1 — Thông tin đề tài")
    first_column, second_column = st.columns(2)
    with first_column:
        st.text_input("Tên đề tài", key="project_title")
        st.text_input("Môn học", key="subject_name")
    with second_column:
        st.text_area("Công nghệ sử dụng", key="technology_stack", height=100)
        st.text_area("Ghi chú thêm", key="additional_notes", height=100)


def _render_file_upload() -> None:
    st.subheader("Bước 2 — Tải lên tệp")
    uploaded = st.file_uploader(
        "Tải lên mã nguồn và tài liệu liên quan",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
        help="Tệp được đọc và xử lý trên máy trước khi gửi từng đoạn nội dung tới LLM.",
    )

    # The keyed uploader keeps its files across reruns, so an empty list means the
    # user removed every file and the stored copies must be cleared as well.
    st.session_state.uploaded_files = [
        {
            "filename": file.name,
            "content": file.getvalue(),
            "size": file.size,
        }
        for file in (uploaded or [])
    ]

    files = st.session_state.uploaded_files
    if not files:
        st.info("Chưa có tệp nào được tải lên.")
        return

    st.write(f"Đã chọn {len(files)} tệp.")
    st.dataframe(
        [
            {
                "Tên tệp": file["filename"],
                "Kích thước (byte)": file.get("size", len(file.get("content", b""))),
                "Định dạng": _extension(file["filename"]),
            }
            for file in files
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_analyze_step() -> None:
    st.subheader("Bước 3 — Phân tích dự án")
    analyze_column, reanalyze_column = st.columns([1, 1])
    with analyze_column:
        analyze_clicked = st.button("Phân tích dự án", type="primary", use_container_width=True)
    with reanalyze_column:
        reanalyze_clicked = st.button("Phân tích lại", use_container_width=True)

    if analyze_clicked or reanalyze_clicked:
        _run_analysis()


def _run_analysis() -> None:
    """Run the core pipeline and commit only successful results to session state."""
    progress = st.progress(0.0)
    status = st.empty()

    def update_progress(message: str, value: float) -> None:
        status.info(message)
        progress.progress(value)

    try:
        result = analyze_project(
            st.session_state.uploaded_files,
            project_title=st.session_state.get("project_title", ""),
            subject_name=st.session_state.get("subject_name", ""),
            technology_stack=st.session_state.get("technology_stack", ""),
            additional_notes=st.session_state.get("additional_notes", ""),
            progress_callback=update_progress,
        )
    except PipelineError as exc:
        _clear_analysis_state()
        st.session_state.file_results = exc.file_results
        st.session_state.analysis_errors = exc.analysis_errors
        st.session_state.analysis_error = str(exc)
        status.error(str(exc))
        return
    except Exception:
        _clear_analysis_state()
        message = "Đã xảy ra lỗi không xác định trong quá trình phân tích."
        st.session_state.analysis_error = message
        status.error(message)
        return

    st.session_state.file_results = result.file_results
    st.session_state.analysis_errors = result.analysis_run.errors
    st.session_state.chunk_analysis = [
        analysis.model_dump(mode="json") for analysis in result.analysis_run.analyses
    ]
    st.session_state.project_knowledge = result.knowledge_result.knowledge.model_dump(mode="json")
    st.session_state.analysis_stats = result.analysis_run.stats.to_dict()
    st.session_state.report = None
    st.session_state.validation_result = None
    st.session_state.analysis_error = None
    st.session_state.knowledge_version += 1
    status.success("Phân tích hoàn tất.")
    progress.progress(1.0)


def _clear_analysis_state() -> None:
    """Drop results of the previous run so they are not shown next to a new error."""
    st.session_state.file_results = []
    st.session_state.analysis_errors = []
    st.session_state.chunk_analysis = []
    st.session_state.project_knowledge = None
    st.session_state.analysis_stats = {}
    st.session_state.report = None
    st.session_state.validation_result = None
    st.session_state.revision_count = 0
    st.session_state.validation_error = None
    st.session_state.knowledge_version += 1


def _render_analysis_results() -> None:
    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    file_results = st.session_state.file_results
    if file_results:
        st.subheader("Trạng thái xử lý tệp")
        st.dataframe(
            [
                {
                    "Tên tệp": result["filename"],
                    "Kích thước (byte)": result["size"],
                    "Định dạng": result["extension"],
                    "Trạng thái": _FILE_STATUS_LABELS.get(result["status"], result["status"]),
                    "Ghi chú": result["error"] or "",
                }
                for result in file_results
            ],
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.analysis_errors:
        with st.expander("Lỗi khi phân tích từng đoạn", expanded=False):
            st.json(st.session_state.analysis_errors)

    stats = st.session_state.analysis_stats
    if stats:
        st.subheader("Thống kê phân tích")
        columns = st.columns(5)
        metrics = [
            ("Tệp đã phân tích", stats.get("files_analyzed", 0)),
            ("Đoạn đã phân tích", stats.get("chunks_analyzed", 0)),
            ("Công nghệ", stats.get("technologies_detected", 0)),
            ("Module", stats.get("modules_detected", 0)),
            ("Chức năng", stats.get("features_detected", 0)),
        ]
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)


def _render_knowledge_review() -> None:
    st.subheader("Bước 4 — Kiểm duyệt tri thức dự án")
    knowledge_data = st.session_state.project_knowledge
    if not knowledge_data:
        st.info("Tri thức dự án sẽ xuất hiện sau khi phân tích hoàn tất.")
        return

    try:
        knowledge = ProjectKnowledge.model_validate(knowledge_data)
    except ValidationError as exc:
        st.error(f"Tri thức dự án hiện tại không hợp lệ: {exc}")
        return

    st.write("Thông tin do hệ thống tổng hợp:")
    st.json(knowledge.model_dump(mode="json"))

    unsupported_features = find_features_without_evidence(knowledge)
    if unsupported_features:
        st.warning(
            "Các chức năng sau chưa có bằng chứng (evidence) nào nhắc tới, nên kiểm tra lại "
            "trước khi sinh báo cáo: " + ", ".join(unsupported_features)
        )

    editor_key = f"knowledge_editor_{st.session_state.knowledge_version}"
    edited_json = st.text_area(
        "Chỉnh sửa tri thức dự án (JSON)",
        value=json.dumps(knowledge.model_dump(mode="json"), ensure_ascii=False, indent=2),
        height=500,
        key=editor_key,
    )
    if st.button("Kiểm tra và lưu tri thức dự án", key="save_knowledge"):
        _save_edited_knowledge(edited_json)

    if st.session_state.chunk_analysis:
        with st.expander("Xem kết quả phân tích từng đoạn (JSON)", expanded=False):
            st.json(st.session_state.chunk_analysis)

    st.caption(
        "Bước sinh báo cáo sẽ sử dụng tri thức dự án đã được người dùng kiểm tra và lưu."
    )


def _render_report_step() -> None:
    """Render report generation and the two requested Markdown views."""
    st.subheader("Bước 5 — Sinh báo cáo")
    if not st.session_state.project_knowledge:
        st.info("Hãy phân tích dự án và kiểm duyệt tri thức dự án trước.")
        return

    if st.button("Sinh báo cáo", type="primary", use_container_width=True):
        try:
            generated = generate_report(
                st.session_state.project_knowledge,
                additional_notes=st.session_state.get("additional_notes", ""),
            )
            st.session_state.report = generated.markdown
            st.session_state.validation_result = None
            st.session_state.revision_count = 0
            st.session_state.validation_error = None
            st.success("Báo cáo đã được sinh.")
        except ReportGenerationError as exc:
            st.error(str(exc))
        except ValidationError as exc:
            st.error(f"Tri thức dự án không hợp lệ: {exc}")

    report = st.session_state.report
    if not report:
        st.info("Chưa có báo cáo. Bấm “Sinh báo cáo” để bắt đầu.")
        return

    _render_download_buttons(report, key_prefix="step5")

    preview_tab, raw_tab, validation_tab = st.tabs(
        ["Xem trước", "Mã Markdown", "Kiểm tra"]
    )
    with preview_tab:
        st.markdown(report)
    with raw_tab:
        st.code(report, language="markdown")
    with validation_tab:
        _render_validation_result()


def _render_download_buttons(report: str, *, key_prefix: str) -> None:
    """Render download buttons for Word (.docx), PDF (.pdf), and Markdown (.md)."""
    project_title = st.session_state.get("project_title", "project")
    docx_col, pdf_col, md_col = st.columns(3)
    with docx_col:
        st.download_button(
            "📄 Tải báo cáo Word (.docx)",
            data=docx_bytes(report),
            file_name=build_report_filename(project_title, "docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
            key=f"{key_prefix}_download_docx",
        )
    with pdf_col:
        st.download_button(
            "📕 Tải báo cáo PDF (.pdf)",
            data=pdf_bytes(report),
            file_name=build_report_filename(project_title, "pdf"),
            mime="application/pdf",
            use_container_width=True,
            key=f"{key_prefix}_download_pdf",
        )
    with md_col:
        st.download_button(
            "📝 Tải báo cáo Markdown (.md)",
            data=markdown_bytes(report),
            file_name=build_report_filename(project_title, "md"),
            mime="text/markdown",
            use_container_width=True,
            key=f"{key_prefix}_download_md",
        )


def _render_validation_step() -> None:
    """Run bounded validation/revision after the report has been generated."""
    st.subheader("Bước 6 — Kiểm tra báo cáo")
    if not st.session_state.report:
        st.info("Hãy sinh báo cáo trước khi kiểm tra.")
        return

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    if st.button("Kiểm tra báo cáo", type="primary", use_container_width=True):
        try:
            with st.spinner("Đang kiểm tra báo cáo và chỉnh sửa nếu cần..."):
                validation_run = validate_and_revise_report(
                    st.session_state.report,
                    st.session_state.project_knowledge,
                )
        except ValidatorError as exc:
            st.session_state.validation_error = str(exc)
            st.error(str(exc))
            return
        except ValidationError as exc:
            message = f"Tri thức dự án không hợp lệ: {exc}"
            st.session_state.validation_error = message
            st.error(message)
            return

        st.session_state.report = validation_run.report
        st.session_state.validation_result = validation_run.validation_result.model_dump(
            mode="json"
        )
        st.session_state.revision_count = validation_run.revision_count
        st.session_state.validation_error = None
        st.rerun()

    _render_validation_result()
    if st.session_state.validation_result:
        st.markdown("#### Tải báo cáo sau khi kiểm tra")
        _render_download_buttons(st.session_state.report, key_prefix="step6")


def _render_validation_result() -> None:
    """Display the Pydantic validation result in the Validation tab/step."""
    validation = st.session_state.validation_result
    if not validation:
        st.info("Chưa chạy kiểm tra.")
        return

    if validation.get("valid"):
        st.success("Hợp lệ — không phát hiện vấn đề.")
    else:
        st.error("Chưa hợp lệ — báo cáo còn vấn đề cần xem xét.")

    st.write(f"Số lần chỉnh sửa tự động: {st.session_state.revision_count}")
    issues = validation.get("issues", [])
    if issues:
        st.dataframe(
            [
                {
                    "Nội dung": issue.get("claim", ""),
                    "Lý do": issue.get("reason", ""),
                    "Mức độ": _SEVERITY_LABELS.get(issue.get("severity"), issue.get("severity")),
                    "Tệp nguồn": issue.get("source_file") or "",
                }
                for issue in issues
            ],
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Kết quả kiểm tra (JSON)", expanded=False):
        st.json(validation)


def _save_edited_knowledge(edited_json: str) -> None:
    """Validate edited JSON and invalidate downstream report state if changed."""
    try:
        parsed = json.loads(edited_json)
        knowledge = ProjectKnowledge.model_validate(parsed)
    except json.JSONDecodeError as exc:
        st.error(f"JSON không hợp lệ: {exc}")
        return
    except ValidationError as exc:
        st.error(f"Tri thức dự án không đúng cấu trúc: {exc}")
        return

    st.session_state.project_knowledge = knowledge.model_dump(mode="json")
    st.session_state.report = None
    st.session_state.validation_result = None
    st.session_state.knowledge_version += 1
    st.success("Tri thức dự án đã được kiểm tra và lưu.")
    st.rerun()


def _extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""


if __name__ == "__main__":
    main()
