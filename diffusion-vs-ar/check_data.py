"""
Script để check format thực sự của data
"""
import sys
sys.path.insert(0, 'src')

from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer

# Load tokenizer
tokenizer = CustomTokenizer.from_pretrained("model_config_tiny")

print("=== Tokenizer Vocabulary ===")
print(f"Vocab size: {len(tokenizer)}")
print(f"\nAll tokens in vocab:")
for i in range(min(50, len(tokenizer))):
    token = tokenizer.decode([i])
    print(f"  {i}: '{token}'")

print("\n=== Special Tokens ===")
print(f"PAD:  {tokenizer.pad_token} = {tokenizer.pad_token_id}")
print(f"MASK: {tokenizer.mask_token} = {tokenizer. mask_token_id}")
print(f"SEP: {tokenizer.sep_token} = {tokenizer.sep_token_id}")
print(f"EOS: {tokenizer.eos_token} = {tokenizer. eos_token_id}")
print(f"BOS: {tokenizer.bos_token} = {tokenizer. bos_token_id}")
print(f"UNK:  {tokenizer.unk_token} = {tokenizer.unk_token_id}")

print("\n=== Test Different Formats ===")
test_inputs = [
    "Sudoku: 003020600",
    "003020600",
    "Sudoku: 003020600",
    " 003020600",
]

for inp in test_inputs: 
    ids = tokenizer.encode(inp)
    decoded = tokenizer.decode(ids)
    print(f"\nInput: '{inp}'")
    print(f"  IDs: {ids}")
    print(f"  Decoded: '{decoded}'")
    print(f"  Has UNK: {4 in ids}")  # 4 seems to be UNK