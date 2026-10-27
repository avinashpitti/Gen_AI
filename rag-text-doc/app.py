import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# --- S3 Configuration ---
S3_BUCKET_NAME = "avinash-rag-hospital-data-2026-521754508486-ap-south-1-an"
S3_FILE_KEY = "hospital-data.txt"
LOCAL_FILE_PATH = "downloaded_hospital_data.txt"

print("Step 0: Fetching document from AWS S3...")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

try:
    s3_client.download_file(S3_BUCKET_NAME, S3_FILE_KEY, LOCAL_FILE_PATH)
    print("Document successfully downloaded from S3!")
except Exception as e:
    print(f"Error downloading file from S3: {e}")
    raise

# 1. Load document
print("Step 1: Loading documents...")
loader = TextLoader(LOCAL_FILE_PATH, encoding="utf-8")
documents = loader.load()

# 2. Split document into chunks
print("Step 2: Splitting documents...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

# 3. Create local embeddings
print("Step 3: Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Initialize Groq LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

# 5. Store in ChromaDB
print("Step 4: Creating vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hospital_information"
)
print("Step 4 completed successfully!")

print("\nHospital Information RAG Chatbot (Powered by AWS S3 & Groq)")
print("Type 'exit' to stop.")

while True:
    question = input("\nAsk a question: ")

    if question.lower().strip() == "exit":
        break

    results = vectorstore.similarity_search(question, k=3)
    context = "\n".join(doc.page_content for doc in results)

    prompt = f"""
    You are a Hospital Data information assistant.

    Answer using ONLY the information in the context.
    If the answer is unavailable, say:
    "Sorry, this disease/information is not available in the Hospital records."

    Context:
    {context}

    Question: {question}

    Answer:
    """

    response = llm.invoke(prompt)
    print("\nAnswer:", response.content)