from llama_cpp import Llama
import time

# Load the quantized GGUF model
llm = Llama(
    model_path="./nil-3b-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

start_time = time.time()
prompt = "System: You are an offline assistant for Africa.\nUser: Explain vector disease mapping.\nAssistant:"

response = llm(
    prompt,
    max_tokens=128,
    stop=["User:", "\n\n"],
    echo=False
)

end_time = time.time()
elapsed = end_time - start_time

text = response["choices"][0]["text"]
tokens = response["usage"]["completion_tokens"]
tps = tokens / elapsed if elapsed > 0 else 0

print("--- NIL-3B INFERENCE OUTPUT ---")
print(text)
print("\n--- PERFORMANCE METRICS ---")
print(f"Generation Time: {elapsed:.2f}s")
print(f"Tokens Generated: {tokens}")
print(f"Speed: {tps:.2f} tokens/sec")