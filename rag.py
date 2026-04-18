import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
loader = TextLoader("data/resa.txt")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
docs_split = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs_split, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=api_key)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """
    You're a strict assistant.

    Rules:
    - Answer ONLY from the context provided
    - If you dont find the answer in context, say: "Not found in the provided context"
    - Keep the answer concise( Max 3-4 lines)

    {context}

    Question:{question}

    Answer:
    """
)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("Ask! (type exit to stop)\n").strip()
    if query.lower() == "exit":
        print("Stopping...")
        break
    if len(query.split()) <= 2:
        print("Write a sentence atleast!\n")
        continue
    response = chain.invoke(query)
    print(f"\nAnswer:{response}\n")
