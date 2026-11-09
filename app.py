"""Streamlit UI for the AI Technical Report Generator."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from pydantic import ValidationError

from core.pipeline import PipelineError, analyze_project
from core.exporter import build_report_filename, markdown_bytes
from core.report_generator import ReportGenerationError, generate_report
from core.validator import ValidatorError, validate_and_revise_report
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
    "sql",
    "html",
    "css",
    "pdf",
    "docx",
    "zip",
]


def main() -> None:
    """Render the application and preserve all pipeline state across reruns."""
    configure_logging()
    _initialize_session_state()
    st.set_page_config(
        page_title="AI Technical Report Generator",
        page_icon="📄",
        layout="wide",
    )

    st.title("AI Technical Report Generator")
    st.caption("Phân tích source code và tài liệu theo pipeline có human review.")

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
    st.subheader("Step 1 — Project Information")
    first_column, second_column = st.columns(2)
    with first_column:
        st.text_input("Project Title", key="project_title")
        st.text_input("Subject Name", key="subject_name")
    with second_column:
        st.text_area("Technology Stack", key="technology_stack", height=100)
        st.text_area("Additional Notes", key="additional_notes", height=100)


def _render_file_upload() -> None:
    st.subheader("Step 2 — Upload Files")
    uploaded = st.file_uploader(
        "Upload source code và tài liệu liên quan",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="file_uploader",
        help="File sẽ được đọc và xử lý cục bộ trước khi gửi từng chunk tới LLM.",
    )

    if uploaded:
        st.session_state.uploaded_files = [
            {
                "filename": file.name,
                "content": file.getvalue(),
                "size": len(file.getvalue()),
            }
            for file in uploaded
        ]

    files = st.session_state.uploaded_files
    if not files:
        st.info("Chưa có file nào được upload.")
        return

    st.write(f"Đã chọn {len(files)} file.")
    st.dataframe(
        [
            {
                "Filename": file["filename"],
                "Size (bytes)": file.get("size", len(file.get("content", b""))),
                "Extension": _extension(file["filename"]),
            }
            for file in files
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_analyze_step() -> None:
    st.subheader("Step 3 — Analyze Project")
    analyze_column, reanalyze_column = st.columns([1, 1])
    with analyze_column:
        analyze_clicked = st.button("Analyze Project", type="primary", use_container_width=True)
    with reanalyze_column:
        reanalyze_clicked = st.button("Re-analyze Project", use_container_width=True)

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
        if exc.file_results:
            st.session_state.file_results = exc.file_results
        if exc.analysis_errors:
            st.session_state.analysis_errors = exc.analysis_errors
        st.session_state.analysis_error = str(exc)
        status.error(str(exc))
        return
    except Exception:
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
    status.success("Analysis completed.")
    progress.progress(1.0)


def _render_analysis_results() -> None:
    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    file_results = st.session_state.file_results
    if file_results:
        st.subheader("File Processing Status")
        st.dataframe(
            [
                {
                    "Filename": result["filename"],
                    "Size (bytes)": result["size"],
                    "Extension": result["extension"],
                    "Status": result["status"],
                    "Error": result["error"] or "",
                }
                for result in file_results
            ],
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.analysis_errors:
        with st.expander("Chunk analysis errors", expanded=False):
            st.json(st.session_state.analysis_errors)

    stats = st.session_state.analysis_stats
    if stats:
        st.subheader("Analysis Metrics")
        columns = st.columns(5)
        metrics = [
            ("Files analyzed", stats.get("files_analyzed", 0)),
            ("Chunks analyzed", stats.get("chunks_analyzed", 0)),
            ("Technologies", stats.get("technologies_detected", 0)),
            ("Modules", stats.get("modules_detected", 0)),
            ("Features", stats.get("features_detected", 0)),
        ]
        for column, (label, value) in zip(columns, metrics):
            column.metric(label, value)


def _render_knowledge_review() -> None:
    st.subheader("Step 4 — Review Project Knowledge")
    knowledge_data = st.session_state.project_knowledge
    if not knowledge_data:
        st.info("Project Knowledge sẽ xuất hiện sau khi Analyze Project hoàn tất.")
        return

    try:
        knowledge = ProjectKnowledge.model_validate(knowledge_data)
    except ValidationError as exc:
        st.error(f"Project Knowledge hiện tại không hợp lệ: {exc}")
        return

    st.write("Thông tin do hệ thống tổng hợp:")
    st.json(knowledge.model_dump(mode="json"))

    editor_key = f"knowledge_editor_{st.session_state.knowledge_version}"
    edited_json = st.text_area(
        "Chỉnh sửa Project Knowledge JSON",
        value=json.dumps(knowledge.model_dump(mode="json"), ensure_ascii=False, indent=2),
        height=500,
        key=editor_key,
    )
    if st.button("Validate and Save Project Knowledge", key="save_knowledge"):
        _save_edited_knowledge(edited_json)

    if st.session_state.chunk_analysis:
        with st.expander("Xem chunk analysis JSON", expanded=False):
            st.json(st.session_state.chunk_analysis)

    st.caption(
        "Bước sinh báo cáo sẽ sử dụng Project Knowledge đã được người dùng kiểm tra và lưu."
    )


def _render_report_step() -> None:
    """Render report generation and the two requested Markdown views."""
    st.subheader("Step 5 — Generate Report")
    if not st.session_state.project_knowledge:
        st.info("Hãy phân tích project và review Project Knowledge trước.")
        return

    if st.button("Generate Report", type="primary", use_container_width=True):
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
            st.error(f"Project Knowledge không hợp lệ: {exc}")

    report = st.session_state.report
    if not report:
        st.info("Chưa có báo cáo. Bấm Generate Report để bắt đầu.")
        return

    preview_tab, raw_tab, validation_tab = st.tabs(
        ["Preview", "Raw Markdown", "Validation"]
    )
    with preview_tab:
        st.markdown(report)
    with raw_tab:
        st.code(report, language="markdown")
        st.download_button(
            "Download Markdown",
            data=markdown_bytes(report),
            file_name=build_report_filename(
                st.session_state.get("project_title", "project")
            ),
            mime="text/markdown",
            use_container_width=True,
        )
    with validation_tab:
        _render_validation_result()


def _render_validation_step() -> None:
    """Run bounded validation/revision after the report has been generated."""
    st.subheader("Step 6 — Validate Report")
    if not st.session_state.report:
        st.info("Hãy sinh báo cáo trước khi validation.")
        return

    if st.session_state.validation_error:
        st.error(st.session_state.validation_error)

    if st.button("Validate Report", type="primary", use_container_width=True):
        try:
            with st.spinner("Đang kiểm tra report và revise nếu cần..."):
                validation_run = validate_and_revise_report(
                    st.session_state.report,
                    st.session_state.project_knowledge,
                )
        except ValidatorError as exc:
            st.session_state.validation_error = str(exc)
            st.error(str(exc))
            return
        except ValidationError as exc:
            message = f"Project Knowledge không hợp lệ: {exc}"
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


def _render_validation_result() -> None:
    """Display the Pydantic validation result in the Validation tab/step."""
    validation = st.session_state.validation_result
    if not validation:
        st.info("Chưa chạy validation.")
        return

    if validation.get("valid"):
        st.success("Valid — không phát hiện issue.")
    else:
        st.error("Invalid — report còn issue cần xem xét.")

    st.write(f"Revision attempts: {st.session_state.revision_count}")
    issues = validation.get("issues", [])
    if issues:
        st.dataframe(issues, use_container_width=True, hide_index=True)
    with st.expander("Validation JSON", expanded=False):
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
        st.error(f"Project Knowledge không đúng schema: {exc}")
        return

    st.session_state.project_knowledge = knowledge.model_dump(mode="json")
    st.session_state.report = None
    st.session_state.validation_result = None
    st.session_state.knowledge_version += 1
    st.success("Project Knowledge đã được validate và lưu.")
    st.rerun()


def _extension(filename: str) -> str:
    return filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""


if __name__ == "__main__":
    main()
