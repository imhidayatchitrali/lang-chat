import os
import openai  # or use Bedrock if required
import tkinter as tk
from tkinter import filedialog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI  # Fixed imports
from langchain_community.vectorstores import FAISS  # Fixed import
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Fixed imports
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

def open_file_dialog():
    """Open file selection dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a Document",
        filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    return file_path

def process_document(file_path):
    """Process and embed the document for retrieval."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def chat_with_document(vectorstore, query):
    """Retrieve and answer a query based on the document."""
    retriever = vectorstore.as_retriever()
    llm = OpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    response = qa_chain.invoke({"query": query})  # Fixed `invoke()` usage
    return response["result"]

if __name__ == "__main__":
    print("Welcome to the Document Q&A CLI!")

    file_path = open_file_dialog()  # Open file picker

    if not file_path:
        print("No file selected. Exiting.")
        exit(1)

    print(f"Selected file: {file_path}")
    print("Processing document... Please wait.")
    vectorstore = process_document(file_path)
    print("Document processed successfully! You can now ask questions.")

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the chat.")
            break

        response = chat_with_document(vectorstore, query)
        print("\nResponse:", response)
