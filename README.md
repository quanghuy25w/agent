# Agent

RAG chatbot for university curriculum consulting.
Stack:
- LlamaIndex
- ChromaDB (persistent)
- Groq API (replacing Ollama)

## Main files
- `agent/NCKH/src/agent/state.py`: runtime config and env loading.
- `agent/NCKH/src/agent/nodes.py`: DOCX parsing, metadata extraction, index build/load.
- `agent/NCKH/src/agent/graph.py`: query engine and response prompt.
- `agent/NCKH/src/agent/run_agent.py`: CLI entrypoint.

## One-time setup
From `D:\NCKH`:

```powershell
agent\NCKH\.venv\Scripts\python.exe -m pip install -r agent\NCKH\requirements.txt
```

## API key setup
Option 1 (current terminal only):

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Option 2 (recommended): create `agent/NCKH/.env` from `.env.example`.

## Run
Build index from DOCX and ask one question:

```powershell
agent\NCKH\.venv\Scripts\python.exe agent\NCKH\src\agent\run_agent.py --rebuild -q "Tong so tin chi nganh Cong nghe thong tin la bao nhieu?"
```

Interactive mode:

```powershell
agent\NCKH\.venv\Scripts\python.exe agent\NCKH\src\agent\run_agent.py
```
Commands in interactive mode:
- `exit` or `quit`: stop.
- `/reset`: clear chat memory.

Adjust retrieval:

```powershell
agent\NCKH\.venv\Scripts\python.exe agent\NCKH\src\agent\run_agent.py --top-k 12
```

Hide source list:

```powershell
agent\NCKH\.venv\Scripts\python.exe agent\NCKH\src\agent\run_agent.py --no-sources
```

## Optional tuning via `.env`
- `GROQ_MODEL=llama-3.1-8b-instant`
- `EMBEDDING_PROVIDER=huggingface`
- `EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder`
- `MAX_HISTORY_TURNS=8`
- `MAX_CONTEXT_CHARS=7000`
