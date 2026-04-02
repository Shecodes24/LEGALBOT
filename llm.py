from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key="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"
)
response = llm.invoke("Hello world")
print(response.content)