from llama_cpp import Llama

print("🔥 Loading NIL-3B...")

# Load the model directly from your local file
llm = Llama(
    model_path="./nil-3b-q4_k_m.gguf",
    n_ctx=2048, # Context window
    verbose=False
)

print("✅ Model loaded. Testing inference...\n")

# The Test Prompt
prompt = "Explain why sovereign offline AI is more important for African startups than relying on Silicon Valley cloud APIs. Keep it punchy."

print(f"USER: {prompt}\n")
print("NIL-3B: ", end="")

# Stream the output token by token just like ChatGPT
stream = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are NIL-3B, a sovereign edge AI built by Nion Technologies. You are highly intelligent, ruthless, and direct."},
        {"role": "user", "content": prompt}
    ],
    stream=True
)

for chunk in stream:
    token = chunk['choices'][0]['delta'].get('content', '')
    print(token, end='', flush=True)

print("\n\n🩸 Test Complete.")