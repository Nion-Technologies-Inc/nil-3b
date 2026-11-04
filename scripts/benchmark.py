import time
from llama_cpp import Llama

def run_benchmark():
    print("Loading nil-3b model into memory...")
    llm = Llama(
        model_path="./nil-3b-q4_k_m.gguf",
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    prompt = "System: You are an offline AI assistant.\nUser: Explain malaria vector mapping in one paragraph.\nAssistant:"

    print("Running inference benchmark...")
    start_time = time.time()
    
    response = llm(
        prompt,
        max_tokens=128,
        stop=["User:", "\n\n"],
        echo=False
    )
    
    elapsed = time.time() - start_time
    tokens = response["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Total Execution Time : {elapsed:.3f} seconds")
    print(f"Tokens Generated     : {tokens}")
    print(f"Generation Speed     : {tps:.2f} tokens/sec")
    print("===================================================\n")

if __name__ == "__main__":
    run_benchmark()