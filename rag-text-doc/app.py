import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

print("Step 1: Loading documents...")
# 1. Load the document
loader = TextLoader("hospital-data.txt", encoding="utf-8")
documents = loader.load()

print("Step 2: Splitting documents...")
# 2. Split the document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunks = splitter.split_documents(documents)

print("Step 3: Loading embedding model...")
# 3. Create embeddings locally
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

try:
    print("Number of chunks:", len(chunks))

    # Test embedding generation separately
    test_vector = embeddings.embed_query(
        "Hospital Information"
    )

    print("Embedding generated successfully")
    print("Embedding dimensions:", len(test_vector))

    # Create the vector database
    print("Step 4: Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hospital_information"
    )

    print("Step 4 completed successfully!")

except Exception as e:
    print("ERROR during Step 4:")
    print(type(e).__name__, str(e))
    raise

print("\nHospital Information RAG Chatbot")
print("Type 'exit' to stop.")
print("Step 6: Chatbot ready!")

while True:
    question = input("\nAsk a question: ")

    if question.lower().strip() == "exit":
        break

    # 5. Retrieve relevant document chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        doc.page_content for doc in results
    )

    # 6. Send retrieved context to Groq
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