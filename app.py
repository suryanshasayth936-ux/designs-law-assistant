"""Screenshot-ready Streamlit UI for the Designs Law Assistant."""
import html

import streamlit as st

import config
from rag_chain import answer_question

st.set_page_config(page_title="Designs Law Assistant", page_icon="⚖️", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1000px; padding-top: 2rem;}
.q-box {background:#f1f5f9; border-left:5px solid #475569; padding:14px 18px; border-radius:8px; margin:10px 0; color:#0f172a;}
.a-box {background:#ecfdf5; border:1px solid #6ee7b7; border-left:5px solid #059669; padding:16px 20px; border-radius:8px; margin:10px 0 18px; color:#064e3b; font-size:1.05rem; line-height:1.5;}
.a-box.nf {background:#fef2f2; border:1px solid #fca5a5; border-left:5px solid #dc2626; color:#7f1d1d;}
.badge {display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.06em; padding:3px 10px; border-radius:999px; margin-bottom:8px; color:#fff; background:#059669;}
.badge.nf {background:#dc2626;}
.meta {font-size:.8rem; color:#475569; margin-bottom:6px;}
.meta b {color:#0f172a;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚖️ Designs Law Assistant")
st.caption("RAG over the CGPDTM *Manual of Designs Practice & Procedure* (IP India) · ChromaDB + Gemini · strict citations or “Not found”")

if "q_input" not in st.session_state:
    st.session_state.q_input = ""
if "result" not in st.session_state:
    st.session_state.result = None


def ask_cb() -> None:
    """Enter in the box or the Ask button: run whatever is currently typed."""
    st.session_state.pending = st.session_state.q_input


def sample_cb(q: str) -> None:
    """Sidebar button: put the question in the box and run it."""
    st.session_state.q_input = q
    st.session_state.pending = q


def run(q: str) -> None:
    q = q.strip()
    if not q:
        return
    with st.spinner("Searching the manual..."):
        try:
            st.session_state.result = answer_question(q)
        except Exception as exc:  # noqa: BLE001
            st.session_state.result = {"error": str(exc)}


with st.sidebar:
    st.header("LMS evaluation questions")
    for i, q in enumerate(config.EVAL_QUESTIONS, 1):
        st.button(f"Q{i}. {q}", key=f"eval{i}", use_container_width=True, on_click=sample_cb, args=(q,))
    st.divider()
    st.caption(f"LLM: `{config.LLM_MODEL}`  \nEmbeddings: `{config.EMBED_MODEL}`  \nRetrieval: top-{config.RETRIEVE_K} chunks")

st.text_input(
    "Ask a question about designs law",
    key="q_input",
    placeholder="e.g. Who can file an application for registration of a design?",
    on_change=ask_cb,
)
st.button("Ask", on_click=ask_cb)

pending = st.session_state.pop("pending", None)
if pending:
    run(pending)

res = st.session_state.result
if res:
    if "error" in res:
        st.error(res["error"])
    else:
        nf = res["answer"] == config.NOT_FOUND
        st.markdown(f"**Question**<div class='q-box'>{html.escape(res['question'])}</div>", unsafe_allow_html=True)
        badge = "<span class='badge nf'>NOT FOUND IN MANUAL</span>" if nf else "<span class='badge'>GROUNDED ANSWER</span>"
        st.markdown(
            f"**Answer**<div class='a-box{' nf' if nf else ''}'>{badge}<br>{html.escape(res['answer'])}</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"📄 Source Reference Chunks ({len(res['docs'])})", expanded=True):
            for n, d in enumerate(res["docs"], 1):
                m = d.metadata
                st.markdown(
                    f"<div class='meta'><b>#{n}</b> · Page <b>{m.get('page')}</b> · {html.escape(str(m.get('chapter')))} · "
                    f"Section: <b>{html.escape(str(m.get('citation')))}</b></div>",
                    unsafe_allow_html=True,
                )
                st.code(d.page_content, language=None, wrap_lines=True)
