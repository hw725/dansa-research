
import sys
import pandas as pd
from pathlib import Path

# Add workspace paths
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pa')

from processor import process_paragraph_file
from scripts.tune_pa_dp import evaluate_pa_output

# Configuration
BEST_PARAMS = {
    "boundary_threshold": 0.639,
    "boundary_bonus_factor": 0.89,
    "shift_penalty_factor": 0.00016,
}

def run_validation(input_file, output_file, gold_file, desc):
    print(f"\n{'='*60}")
    print(f"STARTING: {desc}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Params: {BEST_PARAMS}")
    print(f"{'='*60}\n")
    
    try:
        process_paragraph_file(
            input_file=input_file,
            output_file=output_file,
            embedder_name='bge',
            max_length=180,
            similarity_threshold=0.7,
            max_workers=4,  # Safe for long run
            batch_size=256,
            verbose=False,
            device='cuda',
            use_boundary_model=True,
            enable_refine=True,
            **BEST_PARAMS
        )
        print(f"\n✅ Processing Complete: {desc}")
        
        # Evaluate
        # For pd/test.csv, the file itself contains gold data (원문, 번역문)
        # But evaluate_pa_output expects separate gold file or handles it.
        # Actually evaluate_pa_output compares output xlsx with gold csv/xlsx.
        # We can use the input file as gold file if it has the columns.
        
        print(f"Evaluating {desc}...")
        f1 = evaluate_pa_output(Path(output_file), Path(gold_file))
        print(f"\n🏆 FINAL RESULT [{desc}]: F1 = {f1:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR in {desc}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. Full Test (Original pd/test.csv)
    # Note: pd/test.csv has '문단식별자', '원문', '번역문' so it can serve as gold.
    run_validation(
        input_file='/workspace/datasets/sentenceragraph/test.csv',
        output_file='/workspace/test_results/pa_output_full_optimized.xlsx',
        gold_file='/workspace/datasets/sentenceragraph/test.csv', 
        desc="FULL TEST (All Paragraphs)"
    )
    
    # 2. Exclude 250 Test
    run_validation(
        input_file='/workspace/test_results/pd_test_exclude250.csv',
        output_file='/workspace/test_results/pa_output_exclude250_optimized.xlsx',
        gold_file='/workspace/test_results/pd_test_exclude250.csv',
        desc="EXCLUDE 250 TEST (Filtered)"
    )
