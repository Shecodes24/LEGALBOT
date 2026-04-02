import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


if "messages" not in st.session_state:
    st.session_state["messages"] = []
# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="LawGPT", page_icon="⚖️")
st.title("⚖️ LawGPT – Llama3 Legal Assistant")

st.markdown("""
<style>
div.stButton > button:first-child { background-color: #ffd0d0; }
#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



def reset_conversation():
    st.session_state.messages = []

st.sidebar.button("🗑️ Reset Chat", on_click=reset_conversation)

# --------------------------------------------------
# EMBEDDINGS + VECTOR STORE
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "my_vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    """Convert retrieved documents into plain text"""
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional legal assistant specialized in the Indian Penal Code (IPC).
Answer ONLY using the provided context.
Be precise, factual, and concise."""
    ),
    (
        "human",
        """Context:
{context}

Chat History:
{chat_history}

Question:
{question}"""
    )
])

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",
    temperature=0
)

# --------------------------------------------------
# RAG PIPELINE (FIXED)
# --------------------------------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda _: "\n".join(
            f"User: {m['content']}" if m["role"] == "user"
            else f"Assistant: {m['content']}"
            for m in st.session_state.messages[-4:]
        )
    }
    | prompt
    | llm
)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
user_input = st.chat_input("Ask a legal question (IPC)…")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            response = rag_chain.invoke(user_input)
            answer = response.content
            print(answer)
            st.write(answer)

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
