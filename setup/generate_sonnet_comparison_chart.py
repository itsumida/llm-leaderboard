#!/usr/bin/env python3
"""
Generate minimalistic Sonnet 4.6 vs 4.5 comparison chart with orange tones
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Professional color palette - teal and coral
COLORS = {
    'sonnet_46': '#14B8A6',   # Teal-500 (modern, professional)
    'sonnet_45': '#F59E0B',   # Amber-500 (warm contrast)
    'border_46': '#0D9488',   # Teal-600 (darker border)
    'border_45': '#D97706',   # Amber-600 (darker border)
    'background': '#FFFFFF',
    'text': '#1F2937',        # Gray-800
    'grid': '#F3F4F6'         # Gray-100
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SF Pro Display', 'Inter', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

def minimal_style(ax):
    """Ultra-minimal professional styling"""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.8, color=COLORS['grid'], axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS['text'], length=4, width=1)

def load_data():
    data_path = Path(__file__).parent.parent / 'aggregated_leaderboard.json'
    with open(data_path) as f:
        return json.load(f)

def plot_sonnet_comparison_orange(data, output_dir):
    """Clean bar chart: Sonnet 4.6 vs 4.5 across datasets"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')

    # Win rates by dataset
    datasets = ['MS MARCO\n(Factual)', 'Paul Graham\n(Synthesis)', 'SciFact\n(Scientific)']
    dataset_keys = ['MSMARCO', 'PG', 'SciFact']

    win_rates_46 = [sonnet_46['by_dataset'][k]['win_rate'] * 100 for k in dataset_keys]
    win_rates_45 = [sonnet_45['by_dataset'][k]['win_rate'] * 100 for k in dataset_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')

    x = np.arange(len(datasets))
    width = 0.36

    bars1 = ax.bar(x - width/2, win_rates_46, width, label='Sonnet 4.6',
                   color=COLORS['sonnet_46'], alpha=0.85,
                   edgecolor=COLORS['border_46'], linewidth=1.8)
    bars2 = ax.bar(x + width/2, win_rates_45, width, label='Sonnet 4.5',
                   color=COLORS['sonnet_45'], alpha=0.85,
                   edgecolor=COLORS['border_45'], linewidth=1.8)

    # Value labels - positioned above bars with more spacing
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3.5,
                f'{height:.1f}%', ha='center', va='bottom',
                fontweight='600', fontsize=10, color=COLORS['text'])

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3.5,
                f'{height:.1f}%', ha='center', va='bottom',
                fontweight='600', fontsize=10, color=COLORS['text'])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11, color=COLORS['text'], fontweight='500')
    ax.set_ylabel('Win Rate (%)', fontsize=12, color=COLORS['text'], fontweight='500', labelpad=10)

    minimal_style(ax)
    ax.set_ylim(0, 85)  # Extra space for labels
    ax.legend(loc='upper left', frameon=False, fontsize=11)

    # Add subtle annotations for standout results
    # Highlight where 4.5 wins (SciFact)
    ax.annotate('4.5 wins', xy=(2, win_rates_45[2]), xytext=(2.35, win_rates_45[2] + 10),
                fontsize=9, color=COLORS['border_45'], fontweight='600',
                arrowprops=dict(arrowstyle='->', color=COLORS['border_45'],
                               lw=1.5, alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'sonnet_comparison_orange.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: sonnet_comparison_orange.png")
    plt.close()

def main():
    print("🎨 Generating Sonnet comparison chart (orange theme)...")
    print()

    data = load_data()
    output_dir = Path(__file__).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    plot_sonnet_comparison_orange(data, output_dir)

    print()
    print(f"✨ Chart saved to: {output_dir}")

if __name__ == '__main__':
    main()
