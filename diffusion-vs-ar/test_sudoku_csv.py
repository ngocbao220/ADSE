"""
Final version - Extract solution từ phần target tokens, KHÔNG decode cả sequence
Support CSV format với columns: quizzes, solutions
"""
import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, 'src')

from llmtuner.tuner. core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, FinetuningArguments, DiffusionArguments


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    if stochastic:  
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:  
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index. gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_decoding(x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, noise):
    topk_mode, schedule = decoding_strategy. split("-")
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":  
        rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
    else:  
        raise NotImplementedError
    
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

    if topk_mode. startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic": 
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError

    masked_to_noise = lowest_k_mask
    if isinstance(noise, torch.Tensor):
        xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
    elif isinstance(noise, (int, float)):
        xt = x0.masked_fill(masked_to_noise, noise)
    else:
        raise NotImplementedError
    return xt


class SudokuTester:  
    def __init__(self, model_path, model_config="model_config_tiny", 
                 diffusion_steps=20, decoding_strategy="stochastic0.5-linear", verbose=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.diffusion_steps = diffusion_steps
        self. decoding_strategy = decoding_strategy
        self.verbose = verbose
        
        if verbose:
            print(f"Loading model from {model_path}...")
        
        model_args = ModelArguments(
            model_name_or_path=model_config,
            checkpoint_dir=model_path,
            cache_dir="./cache"
        )
        
        finetuning_args = FinetuningArguments(stage="mdm", finetuning_type="full")
        
        diffusion_args = DiffusionArguments(
            diffusion_steps=diffusion_steps,
            topk_decoding=True,
            token_reweighting=True,
            time_reweighting="linear",
            alpha=0.25,
            gamma=1,
            decoding_strategy=decoding_strategy
        )
        
        self. model, self.tokenizer = load_model_and_tokenizer(
            model_args, finetuning_args, is_trainable=False,
            diffusion_args=diffusion_args, stage="mdm"
        )
        
        self.model = self.model.to(self. device).eval()
        if verbose:
            print("Model loaded!\n")
    
    def generate_samples(self, x, src_mask):
        self.model.eval()
        attention_mask = torch.ones_like(x)
        batch_size = x.size(0)
        init_maskable_mask = maskable_mask = ~src_mask
        
        for t in range(self.diffusion_steps - 1, -1, -1):
            with torch.no_grad():
                if t == self. diffusion_steps - 1:
                    xt = x. masked_fill(maskable_mask, self.tokenizer.mask_token_id)
                
                if self.verbose and ((self.diffusion_steps - t) % 5 == 0 or t == 0):
                    print(f"  Step {self.diffusion_steps - t}/{self.diffusion_steps}")
                
                t_tensor = torch.full((batch_size,), t, device=x.device)
                logits = self. model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch. cat([logits[:,0:1], logits[:,:-1]], dim=1)
                
                scores = torch.log_softmax(logits, dim=-1)
                scores[: ,: ,self.tokenizer.vocab_size:] = -1000
                x0_scores, x0 = scores.max(-1)
                x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])
                
                if t > 0:
                    xt = topk_decoding(x0, x0_scores, self.decoding_strategy,
                                      init_maskable_mask, t, self.diffusion_steps,
                                      self.tokenizer.mask_token_id)
                else:
                    xt = x0
        return xt
    
    def solve(self, problem, cutoff_len=164, show_steps=False):
        """
        Problem: 81 chữ số liền nhau
        """
        numbers = ''. join(c for c in problem if c.isdigit())
        if len(numbers) != 81:
            raise ValueError(f"Need 81 digits, got {len(numbers)}")
        
        if self.verbose and show_steps:
            print(f"Problem: {numbers[:20]}...  (81 digits)")
            self.print_grid(numbers)
        
        # Encode (không có prefix)
        src_ids = self.tokenizer.encode(numbers) + [self.tokenizer.sep_token_id]
        src_len = len(src_ids)
        
        # Pad
        input_ids = src_ids + [self.tokenizer.pad_token_id] * (cutoff_len - src_len)
        input_ids = input_ids[: cutoff_len]
        
        src_mask = torch.zeros(cutoff_len, dtype=torch.bool).to(self.device)
        src_mask[:src_len] = True
        
        x = torch.tensor([input_ids]).to(self.device)
        src_mask = src_mask.unsqueeze(0)
        
        if self.verbose and show_steps:
            print(f"\nSource:  {src_len} tokens")
            print(f"Generating.. .\n")
        
        # Temporarily disable verbose for generate_samples if not showing steps
        orig_verbose = self.verbose
        if not show_steps:
            self.verbose = False
        
        xt = self.generate_samples(x, src_mask)
        
        self.verbose = orig_verbose
        
        # QUAN TRỌNG: Chỉ lấy phần TARGET (sau src_len), bỏ phần source
        target_ids = xt[0, src_len:].cpu().tolist()
        
        # Decode chỉ phần target
        target_decode = self.tokenizer.decode(target_ids, skip_special_tokens=True)
        
        # Extract digits từ target
        solution = ''.join(c for c in target_decode if c.isdigit())
        
        # Take first 81 digits
        solution = solution[:81]
        
        if self.verbose and show_steps:
            print(f"\n{'='*50}")
            print(f"Target output: {target_decode[: 100]}...")
            print(f"Solution: {solution} (len={len(solution)})")
            print(f"{'='*50}\n")
        
        return solution
    
    def print_grid(self, numbers):
        if len(numbers) < 81:
            return
        print()
        for i in range(9):
            print(' '.join(numbers[i*9:(i+1)*9]))
            if i in [2, 5]:  
                print('-' * 17)
        print()
    
    def verify(self, problem, solution):
        prob = ''. join(c for c in problem if c.isdigit())
        sol = ''.join(c for c in solution if c.isdigit())
        
        if len(prob) != 81 or len(sol) != 81:
            return False
        
        # Check that solution matches problem where problem has digits
        for i in range(81):
            if prob[i] != '0' and prob[i] != sol[i]:  
                return False
        
        # Check rows, columns, and 3x3 boxes
        for i in range(9):
            row = [int(sol[i*9+j]) for j in range(9)]
            if len(set(row)) != 9 or sum(row) != 45:
                return False
            col = [int(sol[j*9+i]) for j in range(9)]
            if len(set(col)) != 9 or sum(col) != 45:
                return False
            br, bc = (i//3)*3, (i%3)*3
            box = [int(sol[r*9+c]) for r in range(br,br+3) for c in range(bc,bc+3)]
            if len(set(box)) != 9 or sum(box) != 45:
                return False
        return True


def test_from_csv(tester, csv_path, max_samples=None, save_results=None):
    """
    Test model trên CSV file với columns: quizzes, solutions
    
    Args:
        tester:  SudokuTester instance
        csv_path: Path to CSV file
        max_samples: Maximum number of samples to test (None = all)
        save_results: Path to save detailed results CSV (None = don't save)
    """
    print(f"\n{'='*60}")
    print(f"Loading data from: {csv_path}")
    print(f"{'='*60}\n")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if 'quizzes' not in df.columns or 'solutions' not in df.columns:
        raise ValueError("CSV must have 'quizzes' and 'solutions' columns")
    
    # Limit samples if specified
    if max_samples:
        df = df. head(max_samples)
    
    print(f"Testing on {len(df)} samples...\n")
    
    results = []
    correct = 0
    
    # Test each sample with progress bar
    for idx, row in tqdm(df. iterrows(), total=len(df), desc="Testing"):
        quiz = row['quizzes']
        ground_truth = row['solutions']
        
        try:
            # Solve
            prediction = tester.solve(quiz, show_steps=False)
            
            # Verify against ground truth
            is_valid = tester.verify(quiz, prediction)
            matches_gt = (prediction == ground_truth)
            
            if is_valid and matches_gt: 
                correct += 1
                status = "✅ CORRECT"
            elif is_valid and not matches_gt: 
                status = "⚠️  VALID but different"
            else:
                status = "❌ INVALID"
            
            results.append({
                'index': idx,
                'quiz':  quiz,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'is_valid': is_valid,
                'matches_ground_truth': matches_gt,
                'status': status
            })
            
        except Exception as e:
            results.append({
                'index':  idx,
                'quiz': quiz,
                'ground_truth':  ground_truth,
                'prediction': None,
                'is_valid':  False,
                'matches_ground_truth': False,
                'status': f"❌ ERROR: {str(e)}"
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Correct: {correct} ({100*correct/len(df):.2f}%)")
    print(f"Accuracy: {correct}/{len(df)}")
    print(f"{'='*60}\n")
    
    # Save detailed results if requested
    if save_results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_results, index=False)
        print(f"Detailed results saved to: {save_results}\n")
    
    # Show some examples
    print("Sample results (first 5):")
    for i, res in enumerate(results[:5]):
        print(f"\n{i+1}. {res['status']}")
        if res['prediction']:
            print(f"   Quiz: {res['quiz'][: 30]}...")
            print(f"   Pred: {res['prediction'][:30]}...")
            print(f"   GT:   {res['ground_truth'][: 30]}...")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_config", default="model_config_tiny", help="Model config")
    parser.add_argument("--csv_file", default=None, help="CSV file with quizzes and solutions")
    parser.add_argument("--txt_file", default=None, help="Text file with one sudoku per line")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to test")
    parser.add_argument("--save_results", default=None, help="Path to save results CSV")
    parser.add_argument("--diffusion_steps", type=int, default=20, help="Number of diffusion steps")
    args = parser.parse_args()
    
    # Initialize tester
    tester = SudokuTester(
        args.model_path, 
        args.model_config,
        diffusion_steps=args.diffusion_steps,
        verbose=False  # Disable verbose for batch testing
    )
    
    if args.csv_file:
        # Test from CSV
        results = test_from_csv(
            tester, 
            args.csv_file, 
            max_samples=args.max_samples,
            save_results=args.save_results
        )
        
    elif args.txt_file:
        # Test from text file (one sudoku per line)
        with open(args.txt_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        
        if args.max_samples:
            lines = lines[:args.max_samples]
        
        correct = 0
        print(f"\nTesting on {len(lines)} samples...\n")
        
        for i, prob in enumerate(tqdm(lines, desc="Testing")):
            try:
                sol = tester.solve(prob, show_steps=False)
                if len(sol) == 81 and tester.verify(prob, sol):
                    correct += 1
            except Exception as e:
                print(f"Error on sample {i}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Results: {correct}/{len(lines)} ({100*correct/len(lines):.1f}%)")
        print(f"{'='*60}\n")
        
    else:
        # Test single example
        example = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
        
        tester.verbose = True
        print(f"{'='*60}\nTest Example\n{'='*60}\n")
        sol = tester.solve(example, show_steps=True)
        
        if len(sol) == 81:
            print("Solution grid:")
            tester.print_grid(sol)
        
        valid = tester.verify(example, sol)
        print(f"{'✅ VALID!' if valid else '❌ INVALID'}")


if __name__ == "__main__":
    main()