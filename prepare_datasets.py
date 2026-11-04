import json
import os
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# --- STRICT DEVICE LIMITS ---
TRANS_LIMIT = 180
REASONING_Q = 300
REASONING_A = 400
PAIRS_PER_LANG = 4000  # Will Cap at 4k to prevent RAM overflow during training - Do you here this engineer incase i hire you. 

formatted_data = []

def add_chatml(system, user, assistant):
    formatted_data.append({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    })

# Pan-African Language BCP-47 Codes by Region (Why isnt there a forrum for this shi)
african_languages = [
    # East Africa (F*ck i used to stay here)
    "swh_Latn", "kik_Latn", "lug_Latn", "luo_Latn", "kin_Latn", "run_Latn",
    # Horn of Africa
    "amh_Ethi", "som_Latn", "orm_Ethi",
    # West Africa
    "yor_Latn", "hau_Latn", "ibo_Latn", "wol_Latn", "ewe_Latn", "aka_Latn", "twi_Latn", "bam_Latn", "fon_Latn",
    # Southern Africa (Is i'm here ryt now)
    "zul_Latn", "xho_Latn", "afr_Latn", "sna_Latn", "ssw_Latn", "tsn_Latn", "tso_Latn", "bem_Latn", "nya_Latn", "nso_Latn", "sot_Latn", "umb_Latn",
    # North & Central Africa
    "arb_Arab", "lin_Latn"
]

# Remove duplicates if any (be serious remove when u want to die)
african_languages = list(set(african_languages))

print(f"1. Downloading & Filtering {len(african_languages)} African Languages from NLLB...")

for lang in african_languages:
    # HF NLLB requires alphabetical config names
    alpha_pair = sorted(["eng_Latn", lang])
    config_name = f"{alpha_pair[0]}-{alpha_pair[1]}"
    
    print(f" -> Processing {lang} (Config: {config_name})...")
    
    try:
        # FIX 1: Added trust_remote_code=True to bypass HF security block (Hellen since u don't read code Mrs. COO - This was a fix)
        dataset = load_dataset("allenai/nllb", config_name, split="train", streaming=True, trust_remote_code=True)
        count = 0
        
        for row in dataset:
            if count >= PAIRS_PER_LANG:
                break
                
            eng_text = row['translation']['eng_Latn']
            tgt_text = row['translation'][lang]
            
            # Apply the 180-character hardware latency limit
            if len(eng_text) < TRANS_LIMIT and len(tgt_text) < TRANS_LIMIT:
                # 1. Train English -> Native Language
                add_chatml(
                    "You are NIL-3B, a fast offline translation engine.",
                    f"Translate to {lang}: {eng_text}",
                    tgt_text
                )
                # 2. Train Native Language -> English
                add_chatml(
                    "You are NIL-3B, a fast offline translation engine.",
                    f"Translate to English: {tgt_text}",
                    eng_text
                )
                count += 1
                
        print(f"    ✓ Added {count * 2} bi-directional pairs for {lang}.")
    except Exception as e:
        print(f"    x Skipped {lang} due to error: {e}")

print("\n2. Downloading & Filtering GSM8K (Logic & Math)...")
try:
    # FIX 2: Updated to openai/gsm8k and added trust_remote_code=True (Hugging Face couldn't shut up about this - LOve HF)
    gsm8k = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
    gsm8k_filtered = gsm8k.filter(lambda x: len(x['question']) < REASONING_Q and len(x['answer']) < REASONING_A)
    gsm8k_sampled = gsm8k_filtered.shuffle(seed=42).select(range(min(5000, len(gsm8k_filtered))))
    
    for row in gsm8k_sampled:
        add_chatml(
            "You are NIL-3B. Think logically and solve the math problem.",
            row['question'],
            row['answer']
        )
    print(f"    ✓ Added {len(gsm8k_sampled)} reasoning samples.")
except Exception as e:
    print(f"    x Failed to load GSM8K: {e}")

# Save the master dataset
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n🚀 ALL DONE! Generated data/train.jsonl with {len(formatted_data)} total samples.")