from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os
import sys

# Load environment variables
load_dotenv()

# Get OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in .env file.")
    sys.exit(1)

# Initialize LLM with OpenRouter base URL + API key
llm = ChatOpenAI(
    model="openai/gpt-4o",  # ✅ OpenRouter model name
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",  # ✅ Force OpenRouter endpoint
)

# Define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node function with error handling and fallback
def chat_node(state: ChatState):
    messages = state['messages']
    try:
        response = llm.invoke(messages)
    except Exception as e:
        if "402" in str(e) or "credit" in str(e).lower():
            print("⚠️ Not enough credits for GPT-4o. Switching to gpt-4o-mini...")
            fallback_llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            try:
                response = fallback_llm.invoke(messages)
            except Exception as e2:
                print(f"❌ API Error (fallback failed): {e2}")
                return {"messages": []}
        else:
            print(f"❌ API Error: {e}")
            return {"messages": []}
    return {"messages": [response]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

# Create checkpointer
checkpointer = SqliteSaver(conn=conn)

# Build graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile chatbot
chatbot = graph.compile(checkpointer=checkpointer)
# for message_chunk,metadata in chatbot.stream(
#     {'messages': [HumanMessage(content='What is the recipe to make pasta')]},
#     config={"configurable": {"thread_id": "thread-1"}},
#     stream_mode="messages"
# ):
#  if message_chunk.content:
#      print(message_chunk.content,end=" ",flush=True)
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        print(thread_id)  # just print it
        all_threads.add(thread_id)  # actually store the value
    
    return list(all_threads)
