import re
import json
import os
from sklearn.model_selection import train_test_split

txt_path = "t8.shakespeare.txt"   


with open(txt_path, "r", encoding="utf-8") as f:
    raw = f.read()


start_idx = raw.find("THE SONNETS")
if start_idx == -1:
    raise ValueError("Could not find 'THE SONNETS' in the file.")
sonnets_part = raw[start_idx:]


pattern = r"\n\s*(\d{1,3})\s*\n(.*?)(?=\n\s*\d{1,3}\s*\n|\Z)"
matches = re.findall(pattern, sonnets_part, flags=re.DOTALL)

print(f"Detected sonnets: {len(matches)}")

sonnets = []
for num_str, body in matches:
    num = int(num_str)
    text = body.strip()

    if len(text.splitlines()) < 4:
        continue
    sonnets.append((num, text))

print(f"Usable sonnets: {len(sonnets)}")   

def make_example(text: str):
  
    user_prompt = (
        "Write a Shakespeare-style poem."
    )
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": text.strip()},
        ]
    }

examples = [make_example(text) for (num, text) in sonnets]


train, temp = train_test_split(examples, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

os.makedirs("data_sonnets_t8", exist_ok=True)

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

save_jsonl("data_sonnets_t8/train.jsonl", train)
save_jsonl("data_sonnets_t8/valid.jsonl", valid)
save_jsonl("data_sonnets_t8/test.jsonl", test)

print("Saved JSONL files in data_sonnets_t8/")
