# pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-groq boto3 python-dotenv pypdf sentence-transformers

import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# --- S3 Configuration ---
S3_BUCKET_NAME = "avinash-rag-hospital-data-2026-521754508486-ap-south-1-an"
S3_FILE_KEY = "sample_policy.pdf"
LOCAL_FILE_PATH = "downloaded_doc.pdf"

# Step 0: Download PDF from AWS S3
print("Step 0: Fetching PDF from AWS S3...")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

try:
    s3_client.download_file(S3_BUCKET_NAME, S3_FILE_KEY, LOCAL_FILE_PATH)
    print("PDF successfully downloaded from S3!")
except Exception as e:
    print(f"Error downloading file from S3: {e}")
    raise

# Step 1: Parse and load PDF pages
print("Step 1: Loading PDF pages...")
loader = PyPDFLoader(LOCAL_FILE_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} page(s) from PDF.")

# Step 2: Split text into chunks
print("Step 2: Splitting text into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Generated {len(chunks)} text chunks.")

# Step 3: Local embedding model
print("Step 3: Loading HuggingFace embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Index into ChromaDB
print("Step 4: Storing chunks in Chroma vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_rag_collection"
)
print("Vector database ready!")

# Step 5: Groq LLM Setup
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

# Step 6: Interactive Q&A loop
print("\n" + "=" * 55)
print("PDF Document Q&A Assistant (AWS S3 + LangChain + Groq)")
print("Type 'exit' to quit.")
print("=" * 55 + "\n")

while True:
    question = input("Ask a question: ")

    if question.lower().strip() == "exit":
        break

    results = vectorstore.similarity_search(question, k=3)

    context_blocks = []
    for doc in results:
        page_num = doc.metadata.get("page", 0) + 1
        context_blocks.append(f"[Page {page_num}]: {doc.page_content}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
    You are an accurate Document Q&A assistant.

    Answer the question using ONLY the provided context. If the answer is not contained in the context, state:
    "The requested information is not available in the provided document."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    response = llm.invoke(prompt)
    print(f"\nAnswer: {response.content}\n")
    print("-" * 55)