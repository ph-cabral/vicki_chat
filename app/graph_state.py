from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    user_message: Optional[str]
    retrieved_docs: Optional[str]
    final_response: Optional[str]
    session_id: Optional[str]

