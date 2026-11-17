import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_path = '../results/benchmark_results.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: Benchmark CSV not found at '{csv_path}'. Run the benchmark suite first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df)} benchmark records from {csv_path}")

    # Approach ordering
    approach_order = ['single', 'sm', 'm', 'mnm', 'island']
    df['approach'] = pd.Categorical(df['approach'], categories=approach_order, ordered=True)
    df = df.sort_values('approach')

    # Group by approach and calculate statistics
    grouped = df.groupby('approach', observed=False)
    summary = grouped.agg({
        'total_time': ['mean', 'std'],
        'fitness_time': ['mean', 'std'],
        'kernel_time': ['mean', 'std'],
        'h2d_time': ['mean', 'std'],
        'd2h_time': ['mean', 'std'],
        'communication_time': ['mean', 'std'],
        'best_fitness': ['mean', 'std'],
        'used_cuda': 'first'
    }).reset_index()

    # Flatten column MultiIndex
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]

    print("\n" + "=" * 115)
    print("PARALLEL GA BENCHMARK SUMMARY TABLE (Mean ± Std)")
    print("=" * 115)
    header = f"{'Approach':<12} {'Total Time(s)':<18} {'Fitness(s)':<18} {'Kernel(s)':<16} {'H2D(s)':<14} {'D2H(s)':<14} {'Best Fit':<15} {'CUDA':<6}"
    print(header)
    print("-" * 115)

    for _, row in summary.iterrows():
        total_str = f"{row['total_time_mean']:.4f}±{row['total_time_std']:.4f}"
        fit_str = f"{row['fitness_time_mean']:.4f}±{row['fitness_time_std']:.4f}"
        kern_str = f"{row['kernel_time_mean']:.4f}±{row['kernel_time_std']:.4f}"
        h2d_str = f"{row['h2d_time_mean']:.4f}"
        d2h_str = f"{row['d2h_time_mean']:.4f}"
        best_str = f"{row['best_fitness_mean']:.6f}"
        cuda_str = "Yes" if row['used_cuda_first'] == 1 else "No"
        print(f"{row['approach']:<12} {total_str:<18} {fit_str:<18} {kern_str:<16} {h2d_str:<14} {d2h_str:<14} {best_str:<15} {cuda_str:<6}")

    print("=" * 115)

    # Calculate Speedup and Efficiency relative to Single
    single_time = summary.loc[summary['approach'] == 'single', 'total_time_mean'].values[0]
    summary['speedup'] = single_time / summary['total_time_mean']

    output_graphs_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), 'graphs')
    os.makedirs(output_graphs_dir, exist_ok=True)

    approaches = summary['approach'].tolist()
    palette = ['#4A90E2', '#50E3C2', '#F5A623', '#BD10E0', '#E74C3C']

    # 1. Total Execution Time
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['total_time_mean'], yerr=summary['total_time_std'], capsize=5, color=palette, alpha=0.85, edgecolor='black')
    plt.ylabel('Total Execution Time (seconds)')
    plt.title('Overall Execution Time Comparison (Lower is Better)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '01_total_execution_time.png'), dpi=300)
    plt.close()

    # 2. Fitness Evaluation Time
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['fitness_time_mean'], yerr=summary['fitness_time_std'], capsize=5, color='#34495E', alpha=0.85, edgecolor='black')
    plt.ylabel('Fitness Evaluation Time (seconds)')
    plt.title('Fitness Evaluation Time across Paradigms')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '02_fitness_evaluation_time.png'), dpi=300)
    plt.close()

    # 3. CUDA Kernel Execution Time
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['kernel_time_mean'], color='#16A085', alpha=0.85, edgecolor='black')
    plt.ylabel('CUDA Kernel Time (seconds)')
    plt.title('Actual GPU Kernel Execution Time')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '03_cuda_kernel_time.png'), dpi=300)
    plt.close()

    # 4. Host-to-Device (H2D) Transfer Time
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['h2d_time_mean'], color='#2980B9', alpha=0.85, edgecolor='black')
    plt.ylabel('Host-to-Device Time (seconds)')
    plt.title('Memory Transfer: Host to Device (H2D)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '04_h2d_transfer_time.png'), dpi=300)
    plt.close()

    # 5. Device-to-Host (D2H) Transfer Time
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['d2h_time_mean'], color='#8E44AD', alpha=0.85, edgecolor='black')
    plt.ylabel('Device-to-Host Time (seconds)')
    plt.title('Memory Transfer: Device to Host (D2H)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '05_d2h_transfer_time.png'), dpi=300)
    plt.close()

    # 6. Communication & Coordination Overhead
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['communication_time_mean'], color='#E67E22', alpha=0.85, edgecolor='black')
    plt.ylabel('Coordination / Packing Time (seconds)')
    plt.title('Process & Sub-population Coordination Overhead')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '06_communication_overhead.png'), dpi=300)
    plt.close()

    # 7. Speedup Relative to Single Process Baseline
    plt.figure(figsize=(9, 5))
    bars = plt.bar(approaches, summary['speedup'], color='#27AE60', alpha=0.85, edgecolor='black')
    plt.axhline(y=1.0, color='red', linestyle='--', label='Baseline (Single = 1.0x)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.2f}x", ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Speedup Factor')
    plt.title('Speedup Relative to Single Process (Higher is Better)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '07_speedup_relative_to_single.png'), dpi=300)
    plt.close()

    # 8. Best Fitness Solution Quality
    plt.figure(figsize=(9, 5))
    plt.bar(approaches, summary['best_fitness_mean'], yerr=summary['best_fitness_std'], capsize=5, color='#D35400', alpha=0.85, edgecolor='black')
    plt.ylabel('Best Fitness (Rastrigin - Lower is Better)')
    plt.title('Final Solution Quality Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_graphs_dir, '08_best_fitness_quality.png'), dpi=300)
    plt.close()

    print(f"\nAll 8 research visualizations successfully generated in: {output_graphs_dir}/")

if __name__ == '__main__':
    main()
