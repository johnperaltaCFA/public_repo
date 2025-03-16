import os
import sys
from llama_cpp import Llama

# Suppress unnecessary logs
sys.stderr = open(os.devnull, 'w')

# Load LLM (Ensure the model file exists)
llm = Llama(model_path="Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf", n_gpu_layers=10, verbose=False)

# Store conversation history
conversation_history = []
CONTEXT_WINDOW = 512  # Model's max token limit
MAX_GENERATE_TOKENS = 100  # Prevents exceeding the limit
RESERVED_TOKENS = 50  # Space for system prompt & metadata

def get_token_length(text):
    """Estimates the number of tokens in a given text."""
    return len(text.split())  # Approximate; use a tokenizer for accuracy

def trim_conversation(conversation_history):
    """Trims old conversation history if it exceeds the context window."""
    while get_token_length(" ".join(conversation_history)) > (CONTEXT_WINDOW - MAX_GENERATE_TOKENS - RESERVED_TOKENS):
        conversation_history.pop(0)  # Remove oldest messages

def format_prompt(conversation_history, user_input):
    """Formats the conversation into a structured prompt for AI processing."""
    system_prompt = "<|im_start|>system\nYou are an AI-powered Equity Research Analyst. Provide financial insights.<|im_end|>\n"
    history = "".join(conversation_history)
    user_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n"
    assistant_prompt = "<|im_start|>analyst\n"
    return system_prompt + history + user_prompt + assistant_prompt

while True:
    try:
        user_input = input("Investor: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Trim conversation history
        trim_conversation(conversation_history)

        # Format input with memory
        prompt = format_prompt(conversation_history, user_input)

        # Debugging: Print the actual prompt being fed to the model
        print("\n===== PROMPT SENT TO LLM =====\n")
        print(prompt)
        print("\n===== END OF PROMPT =====\n")

        # Ensure total input tokens + generated tokens don't exceed the context limit
        input_token_count = get_token_length(prompt)
        max_tokens_allowed = CONTEXT_WINDOW - input_token_count - RESERVED_TOKENS

        if max_tokens_allowed <= 0:
            print("Warning: Conversation history is too long. Clearing history to continue.")
            conversation_history = []  # Reset history to avoid overflow
            prompt = format_prompt(conversation_history, user_input)  # Rebuild prompt
            max_tokens_allowed = MAX_GENERATE_TOKENS  # Reset to safe default

        # Adjust generated token count dynamically
        max_tokens_allowed = max(1, min(max_tokens_allowed, MAX_GENERATE_TOKENS))

        # Generate AI response
        response = llm(prompt, max_tokens=max_tokens_allowed, stop=["<|im_end|>"], echo=False)

        if "choices" in response and response["choices"]:
            analyst_reply = response["choices"][0].get("text", "").strip()
        else:
            analyst_reply = "I'm sorry, I couldn't generate a response."

        # Print response
        print(f"Analyst: {analyst_reply}")

        # Store conversation history
        conversation_history.append(f"<|im_start|>user\n{user_input}<|im_end|>\n")
        conversation_history.append(f"<|im_start|>analyst\n{analyst_reply}<|im_end|>\n")

    except Exception as e:
        print(f"Error: {e}")
        break  # Exit safely on unexpected errors
