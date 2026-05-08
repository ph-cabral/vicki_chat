import streamlit as st
import requests

API_URL = "http://chat-agent:8000/chat"
CURRENT_USER_ID = "1"

st.set_page_config(page_title="Viki Chat", layout="centered")
st.title("Viki Chat - Selección de Personal")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"user_{CURRENT_USER_ID}"
if "messages" not in st.session_state:
    st.session_state.messages = []
    try:
        res = requests.get(
            f"http://chat-agent:8000/history/user_{CURRENT_USER_ID}",
            params={"user_id": CURRENT_USER_ID}
        )
        if res.status_code == 200:
            role_map = {"human": "user", "ai": "assistant"}
            for msg in res.json().get("history", []):
                st.session_state.messages.append({
                    "role": role_map.get(msg["role"], msg["role"]),
                    "content": msg["content"]
                })
    except Exception as e:
        st.warning(f"Error cargando historial: {e}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribi tu mensaje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                res = requests.post(API_URL, json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "user_id": CURRENT_USER_ID
                })
                data = res.json()
                answer = data["response"]
                st.session_state.session_id = data["session_id"]
            except Exception as e:
                answer = f"Error de conexion: {e}"
    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
