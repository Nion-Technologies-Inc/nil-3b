import os
from llama_cpp import Llama

# Load model using available CPU cores
num_cores = os.cpu_count() or 4

print("Loading nil-3b into memory...")
llm = Llama(
    model_path="./nil-3b-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=num_cores,
    verbose=False
)

print("\n==========================================")
print("  NIL-3B Offline Terminal Interface")
print("  Type 'exit' or 'quit' to stop.")
print("==========================================\n")

while True:
    try:
        user_input = input("User > ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        
        if not user_input.strip():
            continue

        prompt = f"System: You are an offline AI assistant for Africa.\nUser: {user_input}\nAssistant:"

        response = llm(
            prompt,
            max_tokens=256,
            stop=["User:", "\n\n"],
            echo=False
        )

        output = response["choices"][0]["text"].strip()
        print(f"\nNIL-3B > {output}\n")

    except KeyboardInterrupt:
        print("\nExiting...")
        break