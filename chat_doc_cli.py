# import os
# import openai  # or use Bedrock if required
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.llms import OpenAI  # Can be replaced with Bedrock LLM
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")  
# openai.api_key = openai_api_key

# def process_document(file_path):
#     """Process and embed the document for retrieval."""
#     if file_path.endswith(".pdf"):
#         loader = PyPDFLoader(file_path)
#     else:
#         loader = TextLoader(file_path)

#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(texts, embeddings)
#     return vectorstore

# def chat_with_document(vectorstore, query):
#     """Retrieve and answer a query based on the document."""
#     retriever = vectorstore.as_retriever()
#     llm = OpenAI()
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

#     response = qa_chain.run(query)
#     return response

# if __name__ == "__main__":
#     print("Welcome to the Document Q&A CLI!")
#     file_path = input("Enter the document path (PDF or text file): ").strip()

#     if not os.path.exists(file_path):
#         print("Error: File not found!")
#         exit(1)

#     print("Processing document... Please wait.")
#     vectorstore = process_document(file_path)
#     print("Document processed successfully! You can now ask questions.")

#     while True:
#         query = input("\nEnter your query (or type 'exit' to quit): ").strip()
#         if query.lower() == "exit":
#             print("Exiting the chat.")
#             break

#         response = chat_with_document(vectorstore, query)
#         print("\nResponse:", response)

import os
import openai  # or use Bedrock if required
import tkinter as tk
from tkinter import filedialog
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import OpenAI  # Can be replaced with Bedrock LLM
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

    response = qa_chain.run(query)
    return response

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

