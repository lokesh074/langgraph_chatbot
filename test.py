import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
import uuid
from datetime import datetime
import psycopg2
import psycopg 

# Configuration
DB_URI = "postgresql://xxxx:xxxx8@xxxx:xxxx/langgraph_db"
GROQ_API_KEY = "gsk_xxxxxic5ud0AyLow"

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-20b")

# State Definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat Node
def chat_node(state: ChatState):
    llm = get_llm()
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- FIXED BUILD_CHATBOT FUNCTION ---
@st.cache_resource
def build_chatbot():
    # IMPORTANT: Use 'psycopg.connect' (v3), NOT 'psycopg2.connect'
    # 'autocommit=True' is highly recommended for the checkpointer
    conn = psycopg.connect(DB_URI, autocommit=True)

    # Now passing the correct v3 connection object
    checkpointer = PostgresSaver(conn)
    
    # Setup tables
    checkpointer.setup()

    # Build Graph
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    
    return graph.compile(checkpointer=checkpointer)

# Database functions for user management
def init_user_db():
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Chats table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            thread_id VARCHAR(100) UNIQUE NOT NULL,
            chat_name VARCHAR(200),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def create_user(username: str):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username) VALUES (%s) RETURNING user_id",
            (username,)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        return user_id
    except psycopg2.IntegrityError:
        conn.rollback()
        cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        return cur.fetchone()[0]
    finally:
        cur.close()
        conn.close()

def get_user_chats(user_id: int):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("""
        SELECT chat_id, thread_id, chat_name, created_at, last_updated
        FROM chats
        WHERE user_id = %s
        ORDER BY last_updated DESC
    """, (user_id,))
    chats = cur.fetchall()
    cur.close()
    conn.close()
    return chats

def create_new_chat(user_id: int, chat_name: str = None):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    thread_id = str(uuid.uuid4())
    if not chat_name:
        chat_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    cur.execute("""
        INSERT INTO chats (user_id, thread_id, chat_name)
        VALUES (%s, %s, %s)
        RETURNING chat_id, thread_id
    """, (user_id, thread_id, chat_name))
    chat_id, thread_id = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_id, thread_id

def update_chat_timestamp(thread_id: str):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("""
        UPDATE chats
        SET last_updated = CURRENT_TIMESTAMP
        WHERE thread_id = %s
    """, (thread_id,))
    conn.commit()
    cur.close()
    conn.close()

def delete_chat(chat_id: int):
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()
    cur.execute("DELETE FROM chats WHERE chat_id = %s", (chat_id,))
    conn.commit()
    cur.close()
    conn.close()

# Initialize database
init_user_db()

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

if "username" not in st.session_state:
    st.session_state.username = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# Login Page
if not st.session_state.username:
    st.title("🤖 Welcome to AI Chatbot")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username_input = st.text_input("Username", placeholder="Enter your username")
        if st.button("Login", use_container_width=True, type="primary"):
            if username_input.strip():
                user_id = create_user(username_input.strip())
                st.session_state.username = username_input.strip()
                st.session_state.user_id = user_id
                st.rerun()
            else:
                st.error("Please enter a valid username")
    st.stop()

# Initialize Chatbot
chatbot = build_chatbot()

# Sidebar
with st.sidebar:
    st.title(f"👤 {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        chat_id, thread_id = create_new_chat(st.session_state.user_id)
        st.session_state.current_thread_id = thread_id
        st.rerun()
    st.divider()
    st.subheader("📚 Your Chats")
    chats = get_user_chats(st.session_state.user_id)
    for chat_id, thread_id, chat_name, created_at, last_updated in chats:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"💬 {chat_name}", key=f"chat_{chat_id}", use_container_width=True):
                st.session_state.current_thread_id = thread_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}"):
                delete_chat(chat_id)
                if thread_id == st.session_state.current_thread_id:
                    st.session_state.current_thread_id = None
                st.rerun()

# Main Chat Area
st.title("💬 AI Chatbot")

if not st.session_state.current_thread_id:
    chat_id, thread_id = create_new_chat(st.session_state.user_id)
    st.session_state.current_thread_id = thread_id

config = {"configurable": {"thread_id": st.session_state.current_thread_id}}

# Display History
try:
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
except Exception:
    pass

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                events = chatbot.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
                response = events['messages'][-1].content
                st.write(response)
                update_chat_timestamp(st.session_state.current_thread_id)
            except Exception as e:
                st.error(f"Error: {str(e)}")