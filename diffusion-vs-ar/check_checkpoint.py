# check_training_data.py
import json
import os

# Tìm file training data
possible_paths = [
    "data/cd4_train.jsonl",
    "countdown_train.jsonl",
    "data/countdown. jsonl",
    "countdown. jsonl"
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("⚠️  Không tìm thấy file training data!")
    print("Searched paths:")
    for p in possible_paths:
        print(f"  - {p}")
    exit(1)

print(f"Found training data:  {data_path}\n")
print(f"{'='*60}")
print("TRAINING DATA FORMAT CHECK")
print(f"{'='*60}\n")

with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 5:  # Check first 5 samples
            break
        
        try:
            item = json.loads(line.strip())
            print(f"Sample {i+1}:")
            print(f"  Keys: {list(item.keys())}")
            
            # Check different possible formats
            if 'input' in item and 'output' in item:
                print(f"  Format: input/output")
                print(f"  Input:   '{item['input']}'")
                print(f"  Output:  '{item['output']}'")
                
                # Validate format
                if ',' not in item['input']:
                    print(f"  ⚠️  WARNING: Input missing commas")
                if '=' not in item['output']:
                    print(f"  ⚠️  WARNING: Output missing '='")
                    
            elif 'messages' in item:
                print(f"  Format: messages (ShareGPT)")
                for msg in item['messages']:
                    print(f"    {msg['role']}: '{msg['content']}'")
                    
            elif 'prompt' in item and 'response' in item:
                print(f"  Format: prompt/response")
                print(f"  Prompt:   '{item['prompt']}'")
                print(f"  Response:  '{item['response']}'")
                
            else:
                print(f"  ⚠️  UNKNOWN FORMAT")
                print(f"  Data: {item}")
            
            print()
            
        except json.JSONDecodeError as e:
            print(f"Sample {i+1}: ❌ JSON parsing error:  {e}\n")

# Check dataset_info.json
dataset_info_path = "data/dataset_info.json"
if os.path.exists(dataset_info_path):
    print(f"\n{'='*60}")
    print("DATASET INFO CONFIG")
    print(f"{'='*60}")
    with open(dataset_info_path, 'r') as f:
        config = json.load(f)
    
    import json as json_module
    print(json_module.dumps(config, indent=2))
else:
    print(f"\n⚠️  No dataset_info.json found at {dataset_info_path}")