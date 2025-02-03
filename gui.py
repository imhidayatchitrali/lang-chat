# import os
# import openai
# import tkinter as tk
# from tkinter import filedialog, scrolledtext, messagebox
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, OpenAI  # Fixed imports
# from langchain_community.vectorstores import FAISS  # Fixed import
# from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Fixed imports
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = openai_api_key

# class ChatDocApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Chat with Document")
#         self.root.geometry("600x500")
        
#         # File Selection Button
#         self.file_label = tk.Label(root, text="No file selected", wraplength=400)
#         self.file_label.pack(pady=5)

#         self.select_button = tk.Button(root, text="Select Document", command=self.open_file_dialog)
#         self.select_button.pack(pady=5)

#         # Chat History
#         self.chat_history = scrolledtext.ScrolledText(root, width=70, height=15, state=tk.DISABLED)
#         self.chat_history.pack(pady=10)

#         # User Input Box
#         self.user_input = tk.Entry(root, width=50)
#         self.user_input.pack(pady=5)

#         self.send_button = tk.Button(root, text="Ask", command=self.ask_question)
#         self.send_button.pack(pady=5)

#         # Initialize vectorstore
#         self.vectorstore = None
#         self.file_path = None

#     def open_file_dialog(self):
#         """Open file selection dialog."""
#         file_path = filedialog.askopenfilename(
#             title="Select a Document",
#             filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
#         )
#         if file_path:
#             self.file_path = file_path
#             self.file_label.config(text=f"Selected: {os.path.basename(file_path)}")
#             self.process_document(file_path)

#     def process_document(self, file_path):
#         """Process and embed the document for retrieval."""
#         try:
#             if file_path.endswith(".pdf"):
#                 loader = PyPDFLoader(file_path)
#             else:
#                 loader = TextLoader(file_path)

#             documents = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             texts = text_splitter.split_documents(documents)

#             embeddings = OpenAIEmbeddings()
#             self.vectorstore = FAISS.from_documents(texts, embeddings)
#             messagebox.showinfo("Success", "Document processed successfully!")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to process document: {str(e)}")

#     def ask_question(self):
#         """Retrieve and answer a query based on the document."""
#         if not self.vectorstore:
#             messagebox.showwarning("Warning", "Please upload and process a document first.")
#             return
        
#         query = self.user_input.get().strip()
#         if not query:
#             return

#         try:
#             retriever = self.vectorstore.as_retriever()
#             llm = OpenAI()
#             qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

#             response = qa_chain.invoke({"query": query})  # Fixed `invoke()` usage
#             answer = response["result"]

#             self.update_chat(f"You: {query}\nBot: {answer}\n")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to retrieve answer: {str(e)}")

#     def update_chat(self, text):
#         """Update the chat history box."""
#         self.chat_history.config(state=tk.NORMAL)
#         self.chat_history.insert(tk.END, text + "\n")
#         self.chat_history.config(state=tk.DISABLED)
#         self.chat_history.yview(tk.END)
#         self.user_input.delete(0, tk.END)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ChatDocApp(root)
#     root.mainloop()


import os
import openai
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

class ChatDocApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat with Document")
        self.root.geometry("600x500")
        
        # File Selection Button
        self.file_label = tk.Label(root, text="No file selected", wraplength=400)
        self.file_label.pack(pady=5)

        self.select_button = tk.Button(root, text="Select Document", command=self.open_file_dialog)
        self.select_button.pack(pady=5)

        # Loading Label (Initially Hidden)
        self.loading_label = tk.Label(root, text="", fg="red")
        self.loading_label.pack(pady=5)

        # Chat History
        self.chat_history = scrolledtext.ScrolledText(root, width=70, height=15, state=tk.DISABLED)
        self.chat_history.pack(pady=10)

        # User Input Box
        self.user_input = tk.Entry(root, width=50)
        self.user_input.pack(pady=5)

        self.send_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.send_button.pack(pady=5)

        # Initialize vectorstore
        self.vectorstore = None
        self.file_path = None

    def set_loading(self, message):
        """Show a loading message and disable UI elements."""
        self.loading_label.config(text=message)
        self.root.update_idletasks()  # Refresh UI immediately

    def reset_loading(self):
        """Clear the loading message and enable UI elements."""
        self.loading_label.config(text="")
        self.root.update_idletasks()

    def open_file_dialog(self):
        """Open file selection dialog."""
        file_path = filedialog.askopenfilename(
            title="Select a Document",
            filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Selected: {os.path.basename(file_path)}")
            threading.Thread(target=self.process_document, args=(file_path,)).start()  # Run in separate thread

    def process_document(self, file_path):
        """Process and embed the document for retrieval (runs in a separate thread)."""
        self.set_loading("Processing document... Please wait.")
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(texts, embeddings)

            messagebox.showinfo("Success", "Document processed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process document: {str(e)}")
        finally:
            self.reset_loading()

    def ask_question(self):
        """Retrieve and answer a query based on the document (runs in a separate thread)."""
        if not self.vectorstore:
            messagebox.showwarning("Warning", "Please upload and process a document first.")
            return
        
        query = self.user_input.get().strip()
        if not query:
            return

        threading.Thread(target=self.get_answer, args=(query,)).start()  # Run in separate thread

    def get_answer(self, query):
        """Fetch answer from LLM using LangChain."""
        self.set_loading("Fetching response... Please wait.")
        try:
            retriever = self.vectorstore.as_retriever()
            llm = OpenAI()
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

            response = qa_chain.invoke({"query": query})  # Fixed `invoke()` usage
            answer = response["result"]

            self.update_chat(f"You: {query}\nBot: {answer}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve answer: {str(e)}")
        finally:
            self.reset_loading()

    def update_chat(self, text):
        """Update the chat history box."""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, text + "\n")
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.yview(tk.END)
        self.user_input.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatDocApp(root)
    root.mainloop()
