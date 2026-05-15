from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.memory import save_message, get_history
from app.config import config


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=config.OPENAI_API_KEY
)

SYSTEM_PROMPT = """Sos Viki, una asistente virtual amigable y profesional. 
Respondés en el mismo idioma que el usuario. Sos concisa pero completa."""

def chat(user_message: str, session_id: str, user_id: str) -> str:
    # Guardar mensaje del usuario
    save_message(session_id, user_id, "human", user_message)
    
    # Obtener historial
    history = get_history(session_id, user_id)
    
    # Construir mensajes
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in history:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    
    # Llamar al LLM
    response = llm.invoke(messages)
    ai_content = response.content
    
    # Guardar respuesta
    save_message(session_id, user_id, "ai", ai_content)
    
    return ai_content

