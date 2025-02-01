import os
import openai  # or use Bedrock if required
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import OpenAI  # Can be replaced with Bedrock LLM
from langchain.chains import RetrievalQA
from fastapi import FastAPI, File, UploadFile, Form
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()


def process_document(file_path):
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

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    vectorstore = process_document(file_path)
    return {"message": "File uploaded and processed successfully", "doc_name": file.filename}

@app.post("/chat/")
async def chat_with_document(query: str = Form(...), doc_name: str = Form(...)):
    file_path = f"uploads/{doc_name}"
    vectorstore = process_document(file_path)
    
    retriever = vectorstore.as_retriever()
    llm = OpenAI()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    response = qa_chain.run(query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
