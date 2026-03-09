from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_if_exists() -> None:
    env_path = _project_root() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _maybe_rerun_with_project_venv() -> None:
    venv_python = _project_root() / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return
    cmd = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


_load_dotenv_if_exists()
_maybe_rerun_with_project_venv()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCKH RAG Agent with Groq")
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Single question mode. If omitted, starts interactive chat.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector index from DOCX files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override similarity_top_k at runtime.",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source references in final answer.",
    )
    return parser.parse_args()


def main() -> None:
    if __package__ in {None, ""}:
        sys.path.append(str(Path(__file__).resolve().parents[1]))
        try:
            from agent.graph import NCKHAgent
            from agent.state import AgentConfig
        except Exception as exc:
            print(f"Loi import thu vien: {exc}")
            print("Hay kich hoat .venv va cai lai dependencies tu requirements.txt.")
            raise SystemExit(1)
    else:
        try:
            from .graph import NCKHAgent
            from .state import AgentConfig
        except Exception as exc:
            print(f"Loi import thu vien: {exc}")
            print("Hay kiem tra lai dependencies cua project.")
            raise SystemExit(1)

    args = parse_args()
    try:
        config = AgentConfig.from_env()
    except Exception as exc:
        print(f"Loi cau hinh: {exc}")
        print("Hay kiem tra file .env va bien GROQ_API_KEY.")
        raise SystemExit(1)

    if args.top_k is not None and args.top_k > 0:
        config.similarity_top_k = args.top_k

    try:
        agent = NCKHAgent(config=config, rebuild_index=args.rebuild)
    except Exception as exc:
        print(f"Loi khoi tao agent: {exc}")
        print("Goi y: thu chay voi --rebuild neu ban vua doi embedding model/provider.")
        raise SystemExit(1)

    include_sources = not args.no_sources

    if args.question:
        try:
            print(agent.ask(args.question, include_sources=include_sources))
        except Exception as exc:
            print(f"Loi xu ly cau hoi: {exc}")
            raise SystemExit(1)
        return

    print("NCKH Groq Agent ready. Type 'exit' to quit. Type '/help' for commands.")
    while True:
        try:
            question = input("\nBan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTam biet.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() in {"/help", "help"}:
            print("Lenh: /reset (xoa nho), /help (tro giup), exit (thoat).")
            continue
        if question.lower() in {"/reset", "reset"}:
            agent.reset()
            print("Agent: Da xoa bo nho hoi thoai.")
            continue
        try:
            print(f"Agent: {agent.ask(question, include_sources=include_sources)}")
        except Exception as exc:
            print(f"Agent: Xin loi, he thong gap loi tam thoi ({exc}). Thu lai sau.")


if __name__ == "__main__":
    main()
