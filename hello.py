from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
import time
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Global variables
current_user_id = ""
vectorstore = None
pc = None
index = None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def initialize_pinecone():
    """Initialize Pinecone index"""
    global pc, index
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "pdf-index"
    
    existing_indexes = pc.list_indexes()
    print(f"Existing indexes: {existing_indexes}")
    
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
    return index_name


@tool
def load_pdfs(pdf_paths: str) -> str:
    """Load PDF files and store them in the vector database for the current user.
    
    Args:
        pdf_paths: Comma-separated paths to PDF files to load.
    """
    global current_user_id, vectorstore
    
    if not current_user_id:
        return "Error: No user ID set. Please set user ID first."
    
    try:
        paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]
        
        if not paths:
            return "Error: No valid PDF paths provided."
        
        # Load documents
        docs = []
        for path in paths:
            if not os.path.exists(path):
                return f"Error: File not found: {path}"
            
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        
        # Add user_id to metadata
        for doc in split_docs:
            doc.metadata["user_id"] = current_user_id
        
        # Store in Pinecone
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore.from_documents(
            split_docs,
            embedding=embeddings,
            index_name="pdf-index",
            namespace=current_user_id,
        )
        
        return f"✅ Successfully loaded and stored {len(split_docs)} chunks from {len(paths)} PDF(s) for user '{current_user_id}'"
    
    except Exception as e:
        return f"Error loading PDFs: {str(e)}"


@tool
def search_documents(query: str) -> str:
    """Search through the user's uploaded documents and return relevant information.
    
    Args:
        query: The question or search query to find information in the documents.
    """
    global current_user_id, vectorstore
    
    if not current_user_id:
        return "Error: No user ID set."
    
    try:
        # Initialize vectorstore for retrieval
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore(
            index_name="pdf-index",
            embedding=embeddings,
            namespace=current_user_id,
        )
        
        # Search for relevant documents
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "namespace": current_user_id
            }
        )
        
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found. Please make sure you have uploaded PDFs first."
        
        # Combine document content
        context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Use LLM to generate answer based on context
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
        
        prompt = f"""Based on the following context from the user's documents, please answer the question.
        
Context:
{context}

Question: {query}

Answer:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return f"{response.content}\n\n---\nRetrieved from {len(docs)} document chunk(s)."
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"


@tool
def finish() -> str:
    """End the conversation and exit the RAG agent."""
    return "Ending the session. Goodbye!"


tools = [load_pdfs, search_documents, finish]

model = ChatOpenAI(model="gpt-4-nano").bind_tools(tools)


def rag_agent(state: AgentState) -> AgentState:
    """Main agent function that processes user input and decides which tools to use."""
    global current_user_id
    
    system_prompt = SystemMessage(content=f"""
    You are RAG Assistant, a helpful AI that helps users upload PDF documents and answer questions about them.
    
    Current user: {current_user_id}
    
    CRITICAL Instructions:
    - When the user asks ANY question about documents, ALWAYS use the 'search_documents' tool immediately with their exact question.
    - If the user provides file paths or wants to upload PDFs, use the 'load_pdfs' tool.
    - If the user wants to exit or finish, use the 'finish' tool.
    - NEVER ask the user to upload documents when they ask a question - they may already have documents uploaded.
    - Always try to search first when a question is asked.
    """)
    
    if not state["messages"]:
        # Simple menu at the start - no LLM call needed
        print("\n🤖 RAG Assistant: What would you like to do?")
        print("1. Upload PDFs")
        print("2. Ask questions about your documents")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            user_input = input("\nEnter PDF file paths (comma-separated): ").strip()
            print(f"\n👤 USER: {user_input}")
        elif choice == "2":
            user_input = input("\nWhat's your question? ").strip()
            print(f"\n👤 USER: {user_input}")
        else:
            user_input = choice
            print(f"\n👤 USER: {user_input}")
        
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
    response = model.invoke(all_messages)
    
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Check for finish tool call
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "ending" in message.content.lower() and
            "session" in message.content.lower()):
            return "end"
    
    return "continue"


def print_messages(messages):
    """Print the most recent messages in a readable format."""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# Build the graph
graph = StateGraph(AgentState)

graph.add_node("agent", rag_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_rag_agent():
    """Main function to run the RAG agent."""
    global current_user_id
    
    print("\n===== RAG ASSISTANT =====")
    
    # Get user ID
    current_user_id = input("Enter your user ID: ").strip().lower()
    print(f"✅ User ID set to: {current_user_id}")
    
    # Initialize Pinecone
    print("\n🔧 Initializing Pinecone...")
    initialize_pinecone()
    print("✅ Pinecone initialized")
    
    # Run the agent
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n===== RAG ASSISTANT FINISHED =====")


if __name__ == "__main__":
    run_rag_agent()