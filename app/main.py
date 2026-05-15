from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncpg

from langchain_core.messages import HumanMessage, AIMessage
from app.graph import build_graph
from app.config import config
from app.memory import build_checkpointer

import traceback

app = FastAPI(
    title="Chat CV Agent",
    description="Agente de selección de personal — Basdonax AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# graph = build_graph()
graph = None
db_pool = None


@app.on_event("startup")
async def startup():
    global db_pool, graph
    db_pool = await asyncpg.create_pool(config.DATABASE_URL)
    cp = await build_checkpointer()
    graph = build_graph().compile(checkpointer=cp)


@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None


def user_id_from_session(session_id: str) -> int:
    # session_id = "user_1" -> 1
    try:
        return int(session_id.split("_")[1])
    except:
        return 1


# @app.get("/history/{session_id}")
# def history(session_id: str):
#     msgs = sessions.get(session_id, [])
#     # Excluir el system prompt
#     filtered = [
#         {"role": m["role"], "content": m["content"]}
#         for m in msgs
#         if m["role"] != "system"
#     ]
#     return {"history": filtered}

@app.get("/history/{session_id}")
async def history(session_id: str):
    user_id = user_id_from_session(session_id)
    rows = await db_pool.fetch(
        "SELECT role, content FROM agent.chat_messages "
        "WHERE session_id = $1 AND user_id = $2 "
        "ORDER BY created_at ASC",
        session_id, user_id,
    )
    return {"history": [{"role": r["role"], "content": r["content"]} for r in rows]}



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or "user_1"
    user_id = user_id_from_session(session_id)

    try:
        # Asegurar que la sesión exista
        await db_pool.execute(
            """
            INSERT INTO agent.chat_sessions (session_id, user_id)
            VALUES ($1, $2)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id, user_id
        )

        # Guardar mensaje del usuario
        await db_pool.execute(
            "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
            session_id, user_id, "human", request.message
        )

        # Cargar historial previo
        rows = await db_pool.fetch(
            "SELECT role, content FROM agent.chat_messages WHERE user_id = $1 ORDER BY created_at ASC",
            user_id
        )
        history = []
        for r in rows:
            if r["role"] == "human":
                history.append(HumanMessage(content=r["content"]))
            else:
                history.append(AIMessage(content=r["content"]))

        graph_config = {"configurable": {"thread_id": session_id}}
        initial_state = {
            "messages": history,
            "session_id": session_id,
            "intent": None,
            "user_message": None,
            "retrieved_docs": None,
            "final_response": None,
        }

        result = await graph.ainvoke(initial_state, config=graph_config)
        answer = result["final_response"]

        # Guardar respuesta del asistente
        await db_pool.execute(
            "INSERT INTO agent.chat_messages (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)",
            session_id, user_id, "ai", answer
        )

        return ChatResponse(
            response=answer,
            session_id=session_id,
            intent=result.get("intent"),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat-cv-agent"}



