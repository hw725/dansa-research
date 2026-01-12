#!/usr/bin/env python
"""
Grid search for boundary_style_prior weights.
Excludes the 250 failure cases used for rule derivation to avoid data leakage.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_exclusion_set(failure_analysis_path: str) -> set:
    """Load the 250 failure cases to exclude from testing."""
    df = pd.read_csv(failure_analysis_path)
    exclusion_set = set(zip(df['book_name'], df['paragraph_id']))
    print(f"Loaded {len(exclusion_set)} exclusion cases from rule derivation")
    return exclusion_set


def get_clean_test_paragraphs(test_csv_path: str, exclusion_set: set) -> list:
    """Get test paragraphs excluding the rule derivation cases."""
    df = pd.read_csv(test_csv_path)
    
    # Get unique paragraphs
    all_paragraphs = df.groupby(['book_name', '문단식별자']).first().reset_index()
    all_paragraphs = list(zip(all_paragraphs['book_name'], all_paragraphs['문단식별자']))
    
    # Exclude rule derivation cases
    clean_paragraphs = [p for p in all_paragraphs if (p[0], p[1]) not in exclusion_set]
    
    print(f"Total paragraphs: {len(all_paragraphs)}")
    print(f"After exclusion: {len(clean_paragraphs)}")
    
    return clean_paragraphs


def sample_paragraphs(paragraphs: list, sample_size: int, seed: int) -> list:
    """Sample paragraphs with a fixed seed."""
    import random
    random.seed(seed)
    
    if sample_size >= len(paragraphs):
        return paragraphs
    
    return random.sample(paragraphs, sample_size)


def update_config_weights(config_path: str, weight_terminal: float, weight_continuation: float):
    """Update the boundary_style_prior weights in config."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['pa_selection_params']['boundary_style_prior']['weight_terminal'] = weight_terminal
    config['pa_selection_params']['boundary_style_prior']['weight_continuation'] = weight_continuation
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_pa_evaluation(paragraphs: list, enable_refine: bool = True) -> dict:
    """Run PA evaluation on selected paragraphs and return metrics."""
    from pa.processor import PAProcessor
    from pa.evaluate import evaluate_pa_results
    
    # Load config
    with open('csp_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load test data
    test_df = pd.read_csv('datasets/pa/test.csv')
    
    # Filter to selected paragraphs
    selected_set = set(paragraphs)
    test_df = test_df[test_df.apply(lambda r: (r['book_name'], r['문단식별자']) in selected_set, axis=1)]
    
    # Group by paragraph
    grouped = test_df.groupby(['book_name', '문단식별자'])
    
    # Initialize processor
    processor = PAProcessor(config)
    
    results = []
    for (book_name, para_id), group in grouped:
        gold_src = group['원문'].tolist()
        gold_tgt = group['번역문'].tolist()
        
        # Concatenate for prediction
        src_text = '\n'.join(gold_src)
        tgt_text = '\n'.join(gold_tgt)
        
        # Run PA
        try:
            pred_result = processor.process(src_text, tgt_text, enable_refine=enable_refine)
            pred_src = pred_result.get('aligned_src', [])
            pred_tgt = pred_result.get('aligned_tgt', [])
        except Exception as e:
            pred_src = []
            pred_tgt = []
        
        results.append({
            'book_name': book_name,
            'paragraph_id': para_id,
            'gold_src': gold_src,
            'gold_tgt': gold_tgt,
            'pred_src': pred_src,
            'pred_tgt': pred_tgt
        })
    
    # Calculate metrics
    metrics = evaluate_pa_results(results)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Grid search for boundary_style_prior weights')
    parser.add_argument('--sample-size', type=int, default=1100, 
                        help='Number of paragraphs to sample (default: 1100, ~half of clean set)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--weight-terminal-values', type=str, default='0.006,0.01,0.015,0.02,0.03',
                        help='Comma-separated weight_terminal values to test')
    parser.add_argument('--weight-continuation-values', type=str, default='-0.006,-0.01,-0.015,-0.02,-0.03',
                        help='Comma-separated weight_continuation values to test')
    parser.add_argument('--enable-refine', action='store_true', default=True,
                        help='Enable refinement pass')
    parser.add_argument('--output-dir', type=str, default='test_results/boundary_weight_search',
                        help='Output directory')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation')
    
    args = parser.parse_args()
    
    # Parse weight values
    terminal_values = [float(x) for x in args.weight_terminal_values.split(',')]
    continuation_values = [float(x) for x in args.weight_continuation_values.split(',')]
    
    # Setup paths
    failure_analysis_path = 'test_results/failure_analysis/analysis_cases_detail.csv'
    test_csv_path = 'datasets/pa/test.csv'
    config_path = 'csp_config.json'
    
    # Load exclusion set
    exclusion_set = load_exclusion_set(failure_analysis_path)
    
    # Get clean paragraphs
    clean_paragraphs = get_clean_test_paragraphs(test_csv_path, exclusion_set)
    
    # Sample paragraphs
    sampled = sample_paragraphs(clean_paragraphs, args.sample_size, args.seed)
    print(f"Sampled {len(sampled)} paragraphs for testing")
    
    # Calculate total experiments
    total_experiments = len(terminal_values) * len(continuation_values)
    print(f"\nGrid search configuration:")
    print(f"  weight_terminal: {terminal_values}")
    print(f"  weight_continuation: {continuation_values}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  enable_refine: {args.enable_refine}")
    
    if not args.yes:
        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
    
    # Backup original config
    with open(config_path, 'r', encoding='utf-8') as f:
        original_config = json.load(f)
    
    original_terminal = original_config['pa_selection_params']['boundary_style_prior']['weight_terminal']
    original_continuation = original_config['pa_selection_params']['boundary_style_prior']['weight_continuation']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run grid search
    results = []
    start_time = time.time()
    
    try:
        for i, wt in enumerate(terminal_values):
            for j, wc in enumerate(continuation_values):
                exp_num = i * len(continuation_values) + j + 1
                print(f"\n[{exp_num}/{total_experiments}] weight_terminal={wt}, weight_continuation={wc}")
                
                # Update config
                update_config_weights(config_path, wt, wc)
                
                # Run evaluation
                exp_start = time.time()
                metrics = run_pa_evaluation(sampled, enable_refine=args.enable_refine)
                exp_elapsed = time.time() - exp_start
                
                result = {
                    'weight_terminal': wt,
                    'weight_continuation': wc,
                    'f1_score': metrics.get('micro_f1_tgt_exact', 0),
                    'similarity': metrics.get('mean_similarity', 0),
                    'elapsed_seconds': exp_elapsed
                }
                results.append(result)
                
                print(f"  F1: {result['f1_score']:.4f}, Similarity: {result['similarity']:.4f} ({exp_elapsed:.1f}s)")
    
    finally:
        # Restore original config
        update_config_weights(config_path, original_terminal, original_continuation)
        print(f"\nRestored original weights: terminal={original_terminal}, continuation={original_continuation}")
    
    # Save results
    total_elapsed = time.time() - start_time
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'grid_search_results.csv'), index=False)
    
    # Find best configuration
    best_idx = results_df['f1_score'].idxmax()
    best = results_df.loc[best_idx]
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(sampled),
        'seed': args.seed,
        'excluded_cases': len(exclusion_set),
        'total_experiments': total_experiments,
        'elapsed_seconds': total_elapsed,
        'best_config': {
            'weight_terminal': best['weight_terminal'],
            'weight_continuation': best['weight_continuation'],
            'f1_score': best['f1_score'],
            'similarity': best['similarity']
        },
        'all_results': results
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("Grid Search Complete")
    print(f"{'='*60}")
    print(f"Best configuration:")
    print(f"  weight_terminal: {best['weight_terminal']}")
    print(f"  weight_continuation: {best['weight_continuation']}")
    print(f"  F1 Score: {best['f1_score']:.4f}")
    print(f"  Similarity: {best['similarity']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
