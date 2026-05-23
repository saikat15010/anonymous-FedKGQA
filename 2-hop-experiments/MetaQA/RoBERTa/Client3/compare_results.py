"""
Compare results across different KGE models

This script reads the evaluation results from all KGE models and
creates a comparison table.
"""

import json
import os
from tabulate import tabulate


def load_results(state_dir, kge_model):
    """Load results for a specific KGE model"""
    results_file = os.path.join(state_dir, f'server_test_results_{kge_model}.json')
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def main():
    """Main comparison function"""
    
    print("=" * 80)
    print("FEDERATED KGQA - RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # KGE models to compare
    models = ['transe', 'distmult', 'rotate', 'complex']
    
    # State directories (update these if different)
    state_dirs = {
        'transe': './state_transe',
        'distmult': './state_distmult',
        'rotate': './state_rotate',
        'complex': './state_complex'
    }
    
    # Load results
    all_results = {}
    for model in models:
        state_dir = state_dirs.get(model, f'./state_{model}')
        results = load_results(state_dir, model)
        if results:
            all_results[model] = results
    
    if not all_results:
        print("No results found! Make sure to run evaluations first.")
        return
    
    # Create comparison table - Overall Results
    print("OVERALL RESULTS")
    print("-" * 80)
    
    table_data = []
    headers = ["Model", "Hits@1", "Hits@3", "Hits@10", "MRR", "Total Questions"]
    
    for model in models:
        if model not in all_results:
            continue
        
        results = all_results[model]
        row = [
            f"RoBERTa-{model.upper()}",
            f"{results['hits@1']:.4f}",
            f"{results['hits@3']:.4f}",
            f"{results['hits@10']:.4f}",
            f"{results['mrr']:.4f}",
            results['total']
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print()
    
    # Find best model for each metric
    print("BEST MODELS PER METRIC")
    print("-" * 80)
    
    metrics = ['hits@1', 'hits@3', 'hits@10', 'mrr']
    best_models = {}
    
    for metric in metrics:
        best_model = max(all_results.items(), key=lambda x: x[1][metric])
        best_models[metric] = (best_model[0], best_model[1][metric])
        print(f"  {metric.upper():10s}: RoBERTa-{best_model[0].upper():8s} ({best_model[1][metric]:.4f})")
    
    print()
    
    # Results by hop count
    print("RESULTS BY HOP COUNT")
    print("-" * 80)
    
    for hop in [1, 2]:
        print(f"\n{hop}-Hop Questions:")
        print("-" * 60)
        
        hop_table_data = []
        hop_headers = ["Model", "Hits@1", "MRR", "Total"]
        
        for model in models:
            if model not in all_results:
                continue
            
            results = all_results[model]
            hop_str = str(hop)
            
            if hop_str in results['by_hop']:
                hop_data = results['by_hop'][hop_str]
                row = [
                    f"RoBERTa-{model.upper()}",
                    f"{hop_data['hits@1']:.4f}",
                    f"{hop_data['mrr']:.4f}",
                    hop_data['total']
                ]
                hop_table_data.append(row)
        
        if hop_table_data:
            print(tabulate(hop_table_data, headers=hop_headers, tablefmt='grid'))
    
    print()
    print("=" * 80)
    
    # Save comparison to file
    output_file = 'results_comparison.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED KGQA - RESULTS COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write("\n\n")
        
        f.write("BEST MODELS PER METRIC\n")
        f.write("-" * 80 + "\n")
        for metric in metrics:
            if metric in best_models:
                model, score = best_models[metric]
                f.write(f"  {metric.upper():10s}: RoBERTa-{model.upper():8s} ({score:.4f})\n")
        f.write("\n")
        
        # Add detailed results
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        for model, results in all_results.items():
            f.write(f"\nRoBERTa-{model.upper()}:\n")
            f.write(f"  Total Questions: {results['total']}\n")
            f.write(f"  Hits@1:  {results['hits@1']:.4f}\n")
            f.write(f"  Hits@3:  {results['hits@3']:.4f}\n")
            f.write(f"  Hits@10: {results['hits@10']:.4f}\n")
            f.write(f"  MRR:     {results['mrr']:.4f}\n")
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
        subprocess.check_call(['pip', 'install', 'tabulate'])
        from tabulate import tabulate
    
    main()
