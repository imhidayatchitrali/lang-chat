import PyPDF2
from docx import Document
import openai
import numpy as np
import os

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

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def get_embeddings(texts):
    """Get OpenAI embeddings for multiple texts"""
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return [np.array(item['embedding']) for item in response['data']]

def get_openai_answer(question, context):
    """Get answer from OpenAI's ChatGPT model"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. "
                 "If the answer isn't in the context, say 'I don't know based on the document'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")

    # Get document
    file_path = input("Enter document path (PDF/DOCX/TXT): ").strip()
    text = read_document(file_path)
    chunks = chunk_text(text)
    
    # Get embeddings for all chunks
    print("Processing document...")
    chunk_embeddings = get_embeddings(chunks)

    # Chat interface
    print("\nDocument loaded. Ask questions (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['exit', 'quit']:
            break

        # Get question embedding
        question_embedding = np.array(get_embeddings([question])[0])
        
        # Calculate similarities
        similarities = []
        for chunk_embedding in chunk_embeddings:
            similarity = np.dot(chunk_embedding, question_embedding) / (
                np.linalg.norm(chunk_embedding) * np.linalg.norm(question_embedding)
            )
            similarities.append(similarity)
        
        # Get top 3 most relevant chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        context = " ".join([chunks[i] for i in top_indices])

        # Generate answer
        answer = get_openai_answer(question, context)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()