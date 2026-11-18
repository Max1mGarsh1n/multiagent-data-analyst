from collections import defaultdict, deque
from threading import Lock

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.sql_agent import answer_question
from app.connectors.postgres import CatalogError
from app.contracts.chat import ChatResponse, SourceInfo
from app.contracts.schema import TableDoc
from app.llm.client import LLMSettings
from app.rag.retriever import get_catalog


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    source_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


app = FastAPI(title="Multiagent Data Analyst API", version="0.1.0")

_MAX_SESSION_MESSAGES = 12
_SESSIONS: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=_MAX_SESSION_MESSAGES))
_SESSION_LOCK = Lock()


@app.get("/health")
def health() -> dict[str, str]:
    settings = LLMSettings.from_env()
    return {
        "status": "ok",
        "llm_model": settings.model,
        "llm_base_url": settings.base_url,
    }


@app.get("/sources", response_model=list[SourceInfo])
def list_sources() -> list[SourceInfo]:
    return [SourceInfo(id="demo", title="PostgreSQL · demo")]


@app.get("/sources/{source_id}/catalog", response_model=list[TableDoc])
def catalog(source_id: str) -> list[TableDoc]:
    try:
        return get_catalog(source_id)
    except (ValueError, CatalogError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос должен быть непустой строкой.")

    with _SESSION_LOCK:
        history = list(_SESSIONS[request.session_id])

    try:
        response = answer_question(question, request.source_id, history=history)
    except (ValueError, CatalogError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from None

    with _SESSION_LOCK:
        session_history = _SESSIONS[request.session_id]
        session_history.append({"role": "user", "content": f"Предыдущий вопрос: {question}"})
        if response.sql:
            session_history.append({"role": "assistant", "content": f"Использованный SQL:\n{response.sql}"})

    return response
