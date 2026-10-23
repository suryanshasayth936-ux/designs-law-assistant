"""Interactive CLI for the Designs Law Assistant.

  python ask.py                 # interactive loop
  python ask.py -q "question"   # single question
  python ask.py --eval          # run the 5 LMS questions -> eval_results.md (+ README table)
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import config
from rag_chain import answer_question

README = config.BASE_DIR / "README.md"
START, END = "<!-- EVAL_TABLE_START -->", "<!-- EVAL_TABLE_END -->"


def _short_source(doc) -> str:
    m = doc.metadata
    return f"p.{m.get('page')} · {m.get('chapter')} · {m.get('citation')}"


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def run_eval() -> None:
    rows, details = [], []
    for i, q in enumerate(config.EVAL_QUESTIONS, 1):
        print(f"\nQ{i}: {q}")
        res = answer_question(q)
        print(f"A{i}: {res['answer']}")
        top = "; ".join(sorted({f"p.{d.metadata.get('page')}" for d in res["docs"]}, key=lambda s: int(s[2:])))
        rows.append(f"| {i} | {_cell(q)} | {_cell(res['answer'])} | {top} |")
        details.append((i, q, res))

    table = "| # | Question | Answer | Retrieved pages |\n|---|---|---|---|\n" + "\n".join(rows)
    out = [
        "# Evaluation results — Designs Law Assistant",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · model `{config.LLM_MODEL}` · embeddings `{config.EMBED_MODEL}` · k={config.RETRIEVE_K}_",
        "",
        table,
        "",
        "## Source chunks per question",
    ]
    for i, q, res in details:
        out += ["", f"### Q{i}. {q}", f"**Answer:** {res['answer']}", ""]
        for d in res["docs"]:
            out.append(f"- {_short_source(d)}")
    Path(config.BASE_DIR / "eval_results.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nWrote eval_results.md")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        if START in text and END in text:
            text = re.sub(
                re.escape(START) + r".*?" + re.escape(END),
                lambda _m: f"{START}\n{table}\n{END}",
                text,
                flags=re.DOTALL,
            )
            README.write_text(text, encoding="utf-8")
            print("Updated results table in README.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Designs Law Assistant CLI")
    ap.add_argument("--eval", action="store_true", help="run the 5 LMS evaluation questions")
    ap.add_argument("-q", "--question", help="ask a single question and exit")
    args = ap.parse_args()

    if args.eval:
        return run_eval()
    if args.question:
        res = answer_question(args.question)
        print(res["answer"])
        return

    print("Designs Law Assistant — ask about the Manual of Designs Practice & Procedure (type 'exit' to quit)")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue
        res = answer_question(q)
        print(f"\nAnswer: {res['answer']}")
        print("Sources:")
        for d in res["docs"]:
            print("  -", _short_source(d))


if __name__ == "__main__":
    main()
