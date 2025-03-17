import sys
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress stderr only when necessary (for debugging)
sys.stderr = open(os.devnull, 'w')

# Load PDF and process it
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(docs)

    return doc_chunks

# Initialize Vector Store
def create_vector_store(doc_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(doc_chunks, embedding=embeddings)
    return vector_store

# Set up RAG Retriever
def get_retriever(vector_store):
    return vector_store.as_retriever()

# Load and process the PDF
pdf_path = "equity_research_report.pdf"  # Change this to your actual PDF file path
doc_chunks = load_and_process_pdf(pdf_path)

# Create vector store and retriever
vector_store = create_vector_store(doc_chunks)
retriever = get_retriever(vector_store)

# Set up LlamaCpp Model
llm = LlamaCpp(
    model_path="Dolphin3.0-Llama3.1-8B-F16.gguf",  # Ensure correct path
    n_ctx=4096,
    temperature=0.7,
    n_threads=24,
    verbose=False,
    max_tokens=1000
)

# Conversation Memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True, k=5)

SYSTEM_PROMPT = "<|im_start|>system\nYou are Dolphin, a helpful AI assistant.<|im_end|>\n"

# Format conversation history
def format_history():
    formatted_messages = [SYSTEM_PROMPT]
    
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            formatted_messages.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    return "\n".join(formatted_messages)

# Main chat function with RAG integration
def chat_function(inputs):
    history = format_history()

    # Retrieve relevant document chunks
    relevant_docs = retriever.get_relevant_documents(inputs["input"])
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Create the final prompt
    prompt_text = f"{history}\n<|im_start|>user\n{inputs['input']}\nContext:\n{context}<|im_end|>\n<|im_start|>assistant\n"

    response = llm.predict(prompt_text).strip()  # Use predict() instead of invoke()
    
    memory.save_context({"input": inputs["input"]}, {"output": response})
    
    return response

# Wrap function into a LangChain Runnable
chain = RunnableLambda(chat_function)

# Interactive Chat Loop
while True:
    user_input = input("\n===== You =====\n\n").strip()
    
    if user_input.lower() in ["exit", "quit"]:
        print("\n===== Bye! =====\n")
        del llm
        break
    
    if not user_input:
        print("Chatbot: Please enter a valid message.")
        continue
    
    try:
        response = chain.invoke({"input": user_input})
        print(f"\n===== AI =====\n\n{response}")
    except Exception as e:
        print(f"Chatbot: Error processing input: {e}")
