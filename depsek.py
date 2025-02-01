import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

def read_document(file_path):
    """Reads PDF, DOCX, or TXT files and returns text"""
    text = ""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format")
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def main():
    # Initialize models
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

    # Get document
    file_path = input("Enter document path (PDF/DOCX/TXT): ").strip()
    text = read_document(file_path)
    chunks = chunk_text(text)
    
    # Create embeddings
    chunk_embeddings = embedding_model.encode(chunks)

    # Chat interface
    print("Document loaded. Ask questions (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
            
        # Find relevant chunks
        question_embedding = embedding_model.encode(question)
        similarities = np.dot(chunk_embeddings, question_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
        )
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = " ".join([chunks[i] for i in top_indices])

        # Generate answer
        response = qa_model(
            f"question: {question} context: {context}",
            max_length=200,
            do_sample=True,
            temperature=0.7
        )[0]['generated_text']

        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()