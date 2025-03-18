import sys
import os
import multiprocessing
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.schema import AIMessage, HumanMessage
from langchain.schema.runnable import RunnableLambda

"""https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF/tree/main"""

sys.stderr = open(os.devnull, 'w')

llm = LlamaCpp(
    model_path="Dolphin3.0-Llama3.1-8B-F16.gguf",
    n_ctx=5000,
    temperature=0.7,
    n_gpu_layers=24,
    n_threads=multiprocessing.cpu_count(),
    verbose=False,
    max_tokens=1000
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True, max_length=5)

SYSTEM_PROMPT = "<|im_start|>system\nYou are Dolphin, a helpful AI equity research assistant.<|im_end|>\n"

def format_history(messages):
    formatted_messages = [SYSTEM_PROMPT]
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif isinstance(msg, AIMessage):
            formatted_messages.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    return "\n".join(formatted_messages)

def chat_function(inputs):
    history = format_history(memory.load_memory_variables({})["history"])
    prompt_text = f"{history}\n<|im_start|>user\n{inputs['input']}<|im_end|>\n<|im_start|>assistant\n"
    
    response = llm.invoke(prompt_text).strip()
    
    memory.save_context({"input": inputs["input"]}, {"output": response})
    
    return response

chain = RunnableLambda(chat_function)

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
