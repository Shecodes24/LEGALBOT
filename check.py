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
groq_api_key = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# LOAD EMBEDDINGS + VECTOR STORE
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

Question:
{question}"""
    )
])

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key="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"
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

print("⚖️ Legal Chatbot (IPC)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    response = rag_chain.invoke(user_input)
    print("\nAssistant:", response.content, "\n")
