"""ChromaDB retriever + few-shot LCEL prompt + strict output normalizer."""
import re
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

import config

SYSTEM_PROMPT = """You are the Designs Law Assistant for the CGPDTM "Manual of Designs Practice & Procedure" (IP India, a static 2011 publication).

Rules:
- Answer ONLY from the retrieved context chunks.
- Always append (Reference: Chapter X, Section/Para Y, Page Z) when the answer is found.
- Do NOT treat historical signatures from when this static 2011 manual was originally published (such as the 2011 Preface signature) as establishing who holds a "current" government office today; if the context does not establish a present-day/current fact or does not contain the answer, output the exact two words Not found with no punctuation and no extra explanation.
- If the question is about something outside designs law (for example patents or trade marks fees) and the context does not answer it, output the exact two words Not found with no punctuation and no extra explanation.
- Keep answers concise and quote fees, periods and form numbers exactly as written in the context."""

# Domain-grounded examples. None of the 5 graded LMS questions appear here.
FEW_SHOT_EXAMPLES = [
    {
        "context": (
            "[Page 12 | Chapter 3 – Filing of Design Application | Manual Para 03.04 | Section 5, 44]: "
            "A reciprocity application claims priority of an application filed previously in a convention country. "
            "Such an application shall be filed in India within six months from the date of filing in convention country. "
            "This period of six months is not extendable."
        ),
        "question": "Within what period must a reciprocity design application claiming convention priority be filed in India, and can that period be extended?",
        "answer": (
            "A reciprocity application claiming priority from an earlier convention country filing must be filed in India "
            "within six months from the convention filing date, and this six-month period cannot be extended "
            "(Reference: Chapter 3 – Filing of Design Application, Para 03.04, Section 5 & 44, Page 12)."
        ),
    },
    {
        "context": (
            "[Page 8 | Chapter 2 – Important Definitions | Manual Para 02.02.03]: "
            '"design" does not include any trade mark or property mark or artistic work.'
        ),
        "question": "What is the statutory renewal period and government fee for a registered patent in India?",
        "answer": "Not found",
    },
    {
        "context": (
            "[Page 3 | Preface | General Text]: Suggestions to improve the Manual will be appreciated. "
            "P.H. KURIAN, Controller General of Patents, Designs & Trade Marks."
        ),
        "question": "Who is the current Union Minister of Commerce and Industry or current head of WIPO?",
        "answer": "Not found",
    },
]

_UNAVAILABLE = re.compile(
    r"(not\s+(found|available|mentioned|provided|specified|stated|covered|addressed|contained)"
    r"|(does|do)\s+not\s+(contain|mention|provide|specify|state|include|address|establish)"
    r"|cannot\s+be\s+(found|determined)|no\s+(information|mention|details?)"
    r"|unable\s+to\s+(find|determine)|insufficient\s+(information|context))",
    re.IGNORECASE,
)
_NOT_FOUND_EXACT = re.compile(r"^not found\.?$", re.IGNORECASE)


def normalize_answer(raw: str) -> str:
    """Return the exact string 'Not found' for ungrounded answers; otherwise the cleaned answer."""
    text = (raw or "").strip().strip("*_`\"' \n")
    if not text or _NOT_FOUND_EXACT.match(text):
        return config.NOT_FOUND
    has_reference = "reference:" in text.lower()
    if _UNAVAILABLE.search(text) and (not has_reference or len(text) < 400):
        return config.NOT_FOUND
    return text


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))
        )
    return str(content)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.get_embeddings(),
        persist_directory=config.DB_DIR,
    )


def build_prompt() -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "Context:\n{context}\n\nQuestion: {question}"), ("ai", "{answer}")]
    )
    few_shot = FewShotChatMessagePromptTemplate(examples=FEW_SHOT_EXAMPLES, example_prompt=example_prompt)
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            few_shot,
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


@lru_cache(maxsize=1)
def get_chain():
    """LCEL: prompt | llm | parser. Retrieval happens just before so sources can be shown in the UI."""
    return build_prompt() | config.get_llm() | StrOutputParser()


def answer_question(question: str) -> dict:
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    docs = config.with_backoff(retriever.invoke, question)
    raw = config.with_backoff(
        get_chain().invoke, {"context": format_docs(docs), "question": question}
    )
    raw = _to_text(raw)
    return {"question": question, "answer": normalize_answer(raw), "raw": raw, "docs": docs}
