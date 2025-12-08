#!/usr/bin/env python3
"""
Visualize Leaderboard - Generate PNG visualization from leaderboard JSON
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def create_leaderboard_viz(input_file: str, output_file: str):
    """Create a visual leaderboard from JSON data - matches original leaderboard.png design"""

    # Load leaderboard data
    with open(input_file, 'r') as f:
        data = json.load(f)

    models = data['models']

    # Extract data for visualization
    model_names = [m['name'] for m in models]
    elos = [m['elo'] for m in models]

    # Create figure with sizing matching results.py
    plt.figure(figsize=(10, max(6, len(models) * 0.5)))

    # Create horizontal bar chart with single color (matching original)
    bars = plt.barh(range(len(models)), elos, color='#4C78A8')

    # Customize the plot to match original
    plt.yticks(range(len(models)), model_names)
    plt.xlabel('ELO Rating', fontweight='bold')
    plt.title('LLM Leaderboard - ELO Rankings', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add ELO values on the bars
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        width = bar.get_width()
        plt.text(width + 10, bar.get_y() + bar.get_height()/2,
                f'{elo:.1f}',
                ha='left', va='center', fontsize=9)

    # Add grid matching original style
    plt.grid(axis='x', alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Leaderboard visualization saved to {output_file}")

    plt.close()


def create_detailed_viz(input_file: str, output_file: str):
    """Create a more detailed visualization with metrics"""

    # Load leaderboard data
    with open(input_file, 'r') as f:
        data = json.load(f)

    models = data['models'][:10]  # Top 10

    # Extract data
    model_names = []
    for m in models:
        name = m['name'].replace('anthropic-', '').replace('openai-', '').replace('google-', '')
        name = name.replace('deepseek-', '').replace('qwen-', '')
        name = name.replace('-', ' ').title()
        # Truncate if too long
        if len(name) > 30:
            name = name[:27] + '...'
        model_names.append(name)

    elos = [m['elo'] for m in models]
    overall_scores = [m['metrics']['overall'] for m in models]
    latencies = [m['performance']['avg_latency_ms'] / 1000 for m in models]  # Convert to seconds

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Color scheme
    colors = plt.cm.viridis(range(len(models)))

    y_pos = range(len(models))

    # Plot 1: ELO Ratings
    ax1 = axes[0]
    bars1 = ax1.barh(y_pos, elos, color=colors, edgecolor='black', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(model_names)], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('ELO Rating', fontsize=11, fontweight='bold')
    ax1.set_title('ELO Rankings', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Add values
    for bar, elo in zip(bars1, elos):
        width = bar.get_width()
        ax1.text(width + 5, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Plot 2: Overall Quality Scores
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, overall_scores, color=colors, edgecolor='black', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # No labels, share with first plot
    ax2.invert_yaxis()
    ax2.set_xlabel('Overall Score (1-5)', fontsize=11, fontweight='bold')
    ax2.set_title('Quality Metrics', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 5)
    ax2.grid(axis='x', alpha=0.3)

    # Add values
    for bar, score in zip(bars2, overall_scores):
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Plot 3: Average Latency
    ax3 = axes[2]
    bars3 = ax3.barh(y_pos, latencies, color=colors, edgecolor='black', linewidth=1)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([])  # No labels
    ax3.invert_yaxis()
    ax3.set_xlabel('Avg Latency (seconds)', fontsize=11, fontweight='bold')
    ax3.set_title('Response Speed', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Add values
    for bar, lat in zip(bars3, latencies):
        width = bar.get_width()
        ax3.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                f'{lat:.1f}s', ha='left', va='center', fontsize=8, fontweight='bold')

    # Main title
    fig.suptitle('LLM RAG Leaderboard - Comprehensive Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Detailed visualization saved to {output_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate PNG visualization of leaderboard')
    parser.add_argument('--input', required=True, help='Input leaderboard JSON file')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--detailed', action='store_true', help='Create detailed multi-panel visualization')

    args = parser.parse_args()

    if args.detailed:
        create_detailed_viz(args.input, args.output)
    else:
        create_leaderboard_viz(args.input, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
