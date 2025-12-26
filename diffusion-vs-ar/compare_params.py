"""
Compare Static vs Adaptive Inference Strategies
With detailed metrics:  accuracy, speed, duplicates
"""

import pandas as pd
import numpy as np
from test_sudoku_csv import SudokuTester, test_from_csv
import time

configs = [
    # Baseline
    {"name": "Static", "adaptive": False, "steps": 20},
    
    # Current
    {"name": "Adaptive-Current", "adaptive": True, "intro": 0.7, "conf": 0.5, "min_steps": 0},
    
    # More conservative
    {"name": "Adaptive-Conservative", "adaptive": True, "intro": 0.85, "conf": 0.6, "min_steps": 8},
    
    # With duplicate check
    {"name": "Adaptive-AntiDup", "adaptive":  True, "intro": 0.8, "conf": 0.55, "min_steps": 5, "check_dup":  True},
    
    # Balanced
    {"name": "Adaptive-Balanced", "adaptive":  True, "intro": 0.75, "conf": 0.5, "min_steps": 6},
]

results = []
baseline_steps = None
baseline_time = None

print("="*80)
print(" " * 20 + "ADAPTIVE INFERENCE COMPARISON")
print("="*80)

for cfg in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {cfg['name']}")
    print(f"{'='*80}")
    
    # Initialize tester
    tester = SudokuTester(
        "output/sudoku/mdm-5m-sudoku",
        diffusion_steps=cfg. get("steps", 20),
        verbose=False
    )
    
    # Configure adaptive settings
    if cfg.get("adaptive"):
        tester.intro_thresh = cfg.get("intro", 0.7)
        tester.conf_thresh = cfg. get("conf", 0.5)
        tester.min_steps = cfg.get("min_steps", 0)
        tester.check_duplicate = cfg. get("check_dup", False)
        tester.use_adaptive = True
    else:
        tester.use_adaptive = False
    
    # Run test
    start_time = time.time()
    test_results = test_from_csv(
        tester, 
        "data/sudoku_test_harder.csv", 
        max_samples=1000  # Adjust as needed
    )
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    correct = sum(1 for r in test_results if r. get('matches_ground_truth', False))
    valid = sum(1 for r in test_results if r. get('is_valid', False))
    total = len(test_results)
    
    # Calculate average steps (if available in results)
    avg_steps = None
    steps_std = None
    early_stops = 0
    
    if 'steps_used' in test_results[0]: 
        steps_list = [r['steps_used'] for r in test_results if 'steps_used' in r]
        avg_steps = np.mean(steps_list)
        steps_std = np. std(steps_list)
        
        # Count early stops (less than max steps)
        max_steps = cfg.get("steps", 20)
        early_stops = sum(1 for s in steps_list if s < max_steps)
    else:
        # Fallback:  use max steps
        avg_steps = cfg.get("steps", 20)
        steps_std = 0
    
    # Check for duplicates
    duplicates = 0
    for r in test_results:
        if r. get('prediction'):
            pred = r['prediction']
            for digit in '123456789':
                if pred.count(digit) > 9:
                    duplicates += 1
                    break
    
    # Store baseline for comparison
    if cfg['name'] == "Static":
        baseline_steps = avg_steps
        baseline_time = elapsed_time
    
    # Calculate speedup
    speedup = baseline_steps / avg_steps if baseline_steps and avg_steps else 1. 0
    time_speedup = baseline_time / elapsed_time if baseline_time else 1.0
    
    # Print summary
    print(f"\n{'─'*80}")
    print(f"Results for {cfg['name']}:")
    print(f"{'─'*80}")
    print(f"  Accuracy:       {correct}/{total} ({100*correct/total:.2f}%)")
    print(f"  Valid:         {valid}/{total} ({100*valid/total:.2f}%)")
    print(f"  Duplicates:    {duplicates}/{total} ({100*duplicates/total:.2f}%)")
    print(f"  Avg Steps:     {avg_steps:. 2f} ± {steps_std:.2f}")
    print(f"  Early Stops:   {early_stops}/{total} ({100*early_stops/total:.1f}%)")
    print(f"  Time:           {elapsed_time:.1f}s")
    if baseline_steps and cfg['name'] != "Static":
        print(f"  Speedup:       {speedup:.2f}x (steps), {time_speedup:.2f}x (time)")
        print(f"  Acc Delta:     {100*(correct/total - correct/total):.2f}% vs Static")
    print(f"{'─'*80}")
    
    # Store results
    result_dict = {
        'Config': cfg['name'],
        'Accuracy': f"{100*correct/total:.2f}%",
        'Valid': f"{100*valid/total:.2f}%",
        'Duplicates': f"{100*duplicates/total:. 2f}%",
        'Avg Steps': f"{avg_steps:. 2f}",
        'Speedup': f"{speedup:. 2f}x" if baseline_steps else "baseline",
        'Time (s)': f"{elapsed_time:.1f}",
        'Early Stop %': f"{100*early_stops/total:.1f}%" if avg_steps else "N/A",
    }
    
    # Add config params
    if cfg. get("adaptive"):
        result_dict. update({
            'Intro':  cfg.get('intro', 'N/A'),
            'Conf': cfg. get('conf', 'N/A'),
            'MinSteps': cfg.get('min_steps', 'N/A'),
        })
    else:
        result_dict.update({
            'Intro': 'N/A',
            'Conf': 'N/A',
            'MinSteps':  'N/A',
        })
    
    results.append(result_dict)

# Print comparison table
print(f"\n{'='*80}")
print(" " * 25 + "FINAL COMPARISON TABLE")
print(f"{'='*80}\n")

df = pd.DataFrame(results)

# Reorder columns for better readability
column_order = [
    'Config', 'Accuracy', 'Valid', 'Duplicates', 
    'Avg Steps', 'Speedup', 'Early Stop %', 'Time (s)',
    'Intro', 'Conf', 'MinSteps'
]
df = df[column_order]

print(df.to_string(index=False))

# Print winner
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

# Find best accuracy
best_acc_idx = df['Accuracy'].str.rstrip('%').astype(float).idxmax()
best_acc_config = df.loc[best_acc_idx, 'Config']
best_acc_value = df.loc[best_acc_idx, 'Accuracy']

# Find best speedup (excluding baseline)
df_adaptive = df[df['Speedup'] != 'baseline']
if len(df_adaptive) > 0:
    best_speedup_idx = df_adaptive['Speedup'].str.rstrip('x').astype(float).idxmax()
    best_speedup_config = df_adaptive. loc[best_speedup_idx, 'Config']
    best_speedup_value = df_adaptive.loc[best_speedup_idx, 'Speedup']
else:
    best_speedup_config = "N/A"
    best_speedup_value = "N/A"

# Find best balance (accuracy * speedup)
if len(df_adaptive) > 0:
    df_adaptive['score'] = (
        df_adaptive['Accuracy']. str.rstrip('%').astype(float) * 
        df_adaptive['Speedup'].str.rstrip('x').astype(float)
    )
    best_balance_idx = df_adaptive['score'].idxmax()
    best_balance_config = df_adaptive.loc[best_balance_idx, 'Config']
    best_balance_acc = df_adaptive.loc[best_balance_idx, 'Accuracy']
    best_balance_speedup = df_adaptive.loc[best_balance_idx, 'Speedup']
else: 
    best_balance_config = "N/A"
    best_balance_acc = "N/A"
    best_balance_speedup = "N/A"

print(f"\n🏆 Best Accuracy:  {best_acc_config} ({best_acc_value})")
print(f"⚡ Best Speedup:   {best_speedup_config} ({best_speedup_value})")
print(f"⭐ Best Balance:   {best_balance_config}")
print(f"   └─ Accuracy: {best_balance_acc}, Speedup: {best_balance_speedup}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")
print(f"• For production (best accuracy):     Use '{best_acc_config}'")
print(f"• For speed-critical (fastest):       Use '{best_speedup_config}'")
print(f"• For balanced performance:           Use '{best_balance_config}'")
print(f"{'='*80}\n")

# Save to CSV
output_file = "comparison_results.csv"
df. to_csv(output_file, index=False)
print(f"✅ Results saved to: {output_file}\n")