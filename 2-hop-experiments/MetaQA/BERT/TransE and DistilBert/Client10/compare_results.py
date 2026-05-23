"""
Compare results across different KGE models and Language Models

This script reads the evaluation results from all combinations of KGE models 
and language models, and creates comprehensive comparison tables.
"""

import json
import os
from tabulate import tabulate


def load_results(state_dir, kge_model, lm_model):
    """Load results for a specific KGE model and language model combination"""
    results_file = os.path.join(state_dir, f'server_test_results_{kge_model}_{lm_model}.json')
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("FEDERATED KGQA - COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # KGE models to compare
    kge_models = ['transe', 'distmult', 'rotate', 'complex']
    
    # Language models to compare
    lm_models = ['roberta', 'bert', 'distilbert']
    
    # State directories (update these if different)
    state_dirs = {
        'transe': './state_transe',
        'distmult': './state_distmult',
        'rotate': './state_rotate',
        'complex': './state_complex'
    }
    
    # Load results for all combinations
    all_results = {}
    for kge_model in kge_models:
        for lm_model in lm_models:
            state_dir = state_dirs.get(kge_model, f'./state_{kge_model}')
            results = load_results(state_dir, kge_model, lm_model)
            if results:
                key = f"{kge_model}_{lm_model}"
                all_results[key] = results
    
    if not all_results:
        print("No results found! Make sure to run evaluations first.")
        return
    
    # Create comparison table - Overall Results
    print("OVERALL RESULTS - ALL COMBINATIONS")
    print("-" * 80)
    
    table_data = []
    headers = ["Model Combination", "Hits@1", "Hits@3", "Hits@10", "MRR", "Total Questions"]
    
    for key in sorted(all_results.keys()):
        results = all_results[key]
        kge_model, lm_model = key.split('_', 1)
        
        row = [
            f"{lm_model.upper()}-{kge_model.upper()}",
            f"{results.get('hits@1', 0):.4f}",
            f"{results.get('hits@3', 0):.4f}",
            f"{results.get('hits@10', 0):.4f}",
            f"{results.get('mrr', 0):.4f}",
            results['total']
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()
    
    # Compare by language model
    print("COMPARISON BY LANGUAGE MODEL")
    print("-" * 80)
    
    for lm_model in lm_models:
        print(f"\n{lm_model.upper()} Results:")
        print("-" * 60)
        
        lm_table_data = []
        lm_headers = ["KGE Model", "Hits@1", "Hits@3", "Hits@10", "MRR"]
        
        for kge_model in kge_models:
            key = f"{kge_model}_{lm_model}"
            if key not in all_results:
                continue
            
            results = all_results[key]
            row = [
                kge_model.upper(),
                f"{results.get('hits@1', 0):.4f}",
                f"{results.get('hits@3', 0):.4f}",
                f"{results.get('hits@10', 0):.4f}",
                f"{results.get('mrr', 0):.4f}"
            ]
            lm_table_data.append(row)
        
        if lm_table_data:
            print(tabulate(lm_table_data, headers=lm_headers, tablefmt='grid'))
    
    print()
    
    # Compare by KGE model
    print("COMPARISON BY KGE MODEL")
    print("-" * 80)
    
    for kge_model in kge_models:
        print(f"\n{kge_model.upper()} Results:")
        print("-" * 60)
        
        kge_table_data = []
        kge_headers = ["Language Model", "Hits@1", "Hits@3", "Hits@10", "MRR"]
        
        for lm_model in lm_models:
            key = f"{kge_model}_{lm_model}"
            if key not in all_results:
                continue
            
            results = all_results[key]
            row = [
                lm_model.upper(),
                f"{results.get('hits@1', 0):.4f}",
                f"{results.get('hits@3', 0):.4f}",
                f"{results.get('hits@10', 0):.4f}",
                f"{results.get('mrr', 0):.4f}"
            ]
            kge_table_data.append(row)
        
        if kge_table_data:
            print(tabulate(kge_table_data, headers=kge_headers, tablefmt='grid'))
    
    print()
    
    # Find best model for each metric
    print("BEST MODELS PER METRIC")
    print("-" * 80)
    
    metrics = ['hits@1', 'hits@3', 'hits@10', 'mrr']
    best_models = {}
    
    for metric in metrics:
        best_key = max(all_results.items(), key=lambda x: x[1].get(metric, 0))
        best_models[metric] = (best_key[0], best_key[1].get(metric, 0))
        kge, lm = best_key[0].split('_', 1)
        print(f"  {metric.upper():10s}: {lm.upper()}-{kge.upper():8s} ({best_key[1].get(metric, 0):.4f})")
    
    print()
    
    # Results by hop count
    print("RESULTS BY HOP COUNT")
    print("-" * 80)
    
    for hop in [1, 2]:
        print(f"\n{hop}-Hop Questions:")
        print("-" * 60)
        
        hop_table_data = []
        hop_headers = ["Model Combination", "Hits@1", "MRR", "Total"]
        
        for key in sorted(all_results.keys()):
            results = all_results[key]
            kge_model, lm_model = key.split('_', 1)
            hop_str = str(hop)
            
            if 'by_hop' in results and hop_str in results['by_hop']:
                hop_data = results['by_hop'][hop_str]
                row = [
                    f"{lm_model.upper()}-{kge_model.upper()}",
                    f"{hop_data.get('hits@1', 0):.4f}",
                    f"{hop_data.get('mrr', 0):.4f}",
                    hop_data['total']
                ]
                hop_table_data.append(row)
        
        if hop_table_data:
            print(tabulate(hop_table_data, headers=hop_headers, tablefmt='grid'))
    
    print()
    print("=" * 80)
    
    # Save comparison to file
    output_file = 'results_comparison_all_models.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED KGQA - COMPREHENSIVE RESULTS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL RESULTS - ALL COMBINATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write("\n\n")
        
        f.write("BEST MODELS PER METRIC\n")
        f.write("-" * 80 + "\n")
        for metric in metrics:
            if metric in best_models:
                key, score = best_models[metric]
                kge, lm = key.split('_', 1)
                f.write(f"  {metric.upper():10s}: {lm.upper()}-{kge.upper():8s} ({score:.4f})\n")
        f.write("\n")
        
        # Add detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        for key, results in sorted(all_results.items()):
            kge_model, lm_model = key.split('_', 1)
            f.write(f"\n{lm_model.upper()}-{kge_model.upper()}:\n")
            f.write(f"  Total Questions: {results['total']}\n")
            f.write(f"  Hits@1:  {results.get('hits@1', 0):.4f}\n")
            f.write(f"  Hits@3:  {results.get('hits@3', 0):.4f}\n")
            f.write(f"  Hits@10: {results.get('hits@10', 0):.4f}\n")
            f.write(f"  MRR:     {results.get('mrr', 0):.4f}\n")
            f.write(f"  No Answer: {results.get('no_answer', 0)}\n")
            f.write(f"  Entity Not Found: {results.get('entity_not_found', 0)}\n")
    
    print(f"Detailed comparison saved to: {output_file}")


if __name__ == '__main__':
    # Check if tabulate is installed
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate package...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tabulate', '--break-system-packages'])
        from tabulate import tabulate
    
    main()
