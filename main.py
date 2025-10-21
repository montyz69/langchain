import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec

# ---- Setup API Keys ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-index"

# Check if index exists
existing_indexes = pc.list_indexes()
print((existing_indexes))
index_exists = any(idx.get("name") == index_name for idx in existing_indexes.get("indexes", []))

if not index_exists:
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("⏳ Waiting for index to be ready...")
    time.sleep(5)  
index = pc.Index(index_name)

# ---- Helper functions ----
def load_pdfs(paths):
    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def store_user_docs(user_id, docs):
    for doc in docs:
        doc.metadata["user_id"] = user_id 

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index_name=index_name,
        namespace=user_id,
    )

    print(f"✅ Stored {len(docs)} chunks for user {user_id} in namespace '{user_id}'")
    return vectorstore

def get_user_qa(user_id):
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=user_id,
    )
    print(vectorstore.index.describe_index_stats())

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "namespace" : user_id
        }
    )
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa

# ---- Main ----
if __name__ == "__main__":
    USER_ID = input("Enter user ID: ").strip().lower()

    while True:
        pdf_input = input("\nEnter PDF file paths (comma-separated) or 'done': ").strip()
        if pdf_input.lower() == "done":
            break
        paths = [p.strip() for p in pdf_input.split(",") if p.strip()]
        docs = load_pdfs(paths)
        store_user_docs(USER_ID, docs)

    time.sleep(5)
    qa = get_user_qa(USER_ID)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        answer = qa.invoke({"query": query})
        print(f"\n💬 Answer: {answer['result']}")
