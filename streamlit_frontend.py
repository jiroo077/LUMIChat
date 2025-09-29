import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

st.set_page_config(page_title="LumiChat", page_icon="💬", layout="centered")

# --- Custom CSS to center title and move it slightly downward ---
st.markdown(
    """
    <style>
    .title-container {
        text-align: center;
        margin-top: 50px;  /* Adjust this to move title downward */
        font-size: 48px;
        font-weight: bold;
        color: #0B3D91;  /* deep blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Display centered title ---
st.markdown('<div class="title-container">🤖 LumiChat</div>', unsafe_allow_html=True)

# --- Initialize session state ---
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# --- Display chat history ---
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
user_input = st.chat_input("Type here...")

if user_input:
    # Add user message to history and display it
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    
    ai_message = response['messages'][-1].content

    # Store AI message in history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
