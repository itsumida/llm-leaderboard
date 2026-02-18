#!/usr/bin/env python3
"""
Generate minimalistic plots for Sonnet 4.6 vs 4.5 comparison blog post
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Minimalistic color palette
COLORS = {
    'sonnet_46': '#7C3AED',   # Purple
    'sonnet_45': '#94A3B8',   # Gray
    'opus_46': '#9333EA',     # Dark purple
    'gpt_51': '#3B82F6',      # Blue
    'accent': '#F59E0B',      # Amber
    'grid': '#F1F5F9'
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
    """Ultra-minimal styling"""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color=COLORS['grid'], axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(colors='#64748B', length=4, width=1)

def load_data():
    data_path = Path(__file__).parent.parent / 'aggregated_leaderboard.json'
    with open(data_path) as f:
        return json.load(f)

def plot_sonnet_comparison(data, output_dir):
    """Main comparison: Sonnet 4.6 vs 4.5"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')

    # Dataset ELOs
    datasets = ['Factual QA', 'Long-form synthesis', 'Scientific verification']
    elos_46 = [
        sonnet_46['by_dataset']['MSMARCO']['elo'],
        sonnet_46['by_dataset']['PG']['elo'],
        sonnet_46['by_dataset']['SciFact']['elo']
    ]
    elos_45 = [
        sonnet_45['by_dataset']['MSMARCO']['elo'],
        sonnet_45['by_dataset']['PG']['elo'],
        sonnet_45['by_dataset']['SciFact']['elo']
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, elos_46, width, label='Sonnet 4.6',
                   color=COLORS['sonnet_46'], alpha=0.9,
                   edgecolor='#6B21A8', linewidth=2.5)
    bars2 = ax.bar(x + width/2, elos_45, width, label='Sonnet 4.5',
                   color=COLORS['sonnet_45'], alpha=0.85,
                   edgecolor='#78716C', linewidth=2.5)

    # Value labels - positioned well above bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 30,
                f'{height:.0f}', ha='center', va='bottom',
                fontweight='700', fontsize=12, color='#1E293B')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 30,
                f'{height:.0f}', ha='center', va='bottom',
                fontweight='700', fontsize=12, color='#1E293B')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13, color='#334155', fontweight='600')
    ax.set_ylabel('ELO Rating', fontsize=14, color='#334155', fontweight='600', labelpad=12)

    minimal_style(ax)
    ax.legend(loc='upper center', frameon=False, fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(1200, 1850)

    plt.tight_layout()
    plt.savefig(output_dir / 'sonnet_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: sonnet_comparison.png")
    plt.close()

def plot_overall_top5(data, output_dir):
    """Top 5 models - horizontal bar"""
    models = sorted(data, key=lambda x: x['overall']['elo'], reverse=True)[:5]

    names = [m['name'].split('-')[-1].title().replace('Claude ', '').replace('Openai ', '')
             .replace('X Ai ', '').replace('Google ', '') for m in models]
    names = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Grok 4 Fast', 'Gemini 3 Flash']
    elos = [m['overall']['elo'] for m in models]

    colors_list = [COLORS['opus_46'], COLORS['gpt_51'], COLORS['sonnet_46'],
                   '#10B981', '#EF4444']

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, elos, color=colors_list, alpha=0.9, height=0.6, edgecolor='none')

    # Value labels
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        ax.text(bar.get_width() - 50, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f}', ha='right', va='center', color='white',
                fontweight='600', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, color='#334155')
    ax.invert_yaxis()
    ax.set_xlabel('ELO Rating', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_xlim(1500, 1850)

    plt.tight_layout()
    plt.savefig(output_dir / 'top5_overall.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: top5_overall.png")
    plt.close()

def plot_sonnet_45_strength(data, output_dir):
    """Highlight Sonnet 4.5's SciFact dominance"""
    models_to_show = ['anthropic-claude-sonnet-4.5', 'anthropic-claude-sonnet-4.6',
                      'anthropic-claude-opus-4-6', 'openai-gpt-5.1']

    models = [m for m in data if m['name'] in models_to_show]
    models = sorted(models, key=lambda x: x['by_dataset']['SciFact']['elo'], reverse=True)

    names = ['Sonnet 4.5', 'Opus 4.6', 'GPT-5.1', 'Sonnet 4.6']
    elos = [m['by_dataset']['SciFact']['elo'] for m in models]

    colors_list = [COLORS['sonnet_45'], COLORS['opus_46'], COLORS['gpt_51'],
                   COLORS['sonnet_46']]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='white')

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, elos, color=colors_list, alpha=0.9, height=0.55, edgecolor='none')

    # Value labels
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f}', ha='left', va='center',
                fontweight='600' if i == 0 else '500',
                fontsize=10, color='#1E293B')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11, color='#334155')
    ax.invert_yaxis()
    ax.set_xlabel('ELO Rating (SciFact)', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_xlim(1550, 1800)

    plt.tight_layout()
    plt.savefig(output_dir / 'scifact_ranking.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: scifact_ranking.png")
    plt.close()

def plot_win_rate_comparison(data, output_dir):
    """Win rate comparison scatter"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')
    opus_46 = next(m for m in data if m['name'] == 'anthropic-claude-opus-4-6')
    gpt_51 = next(m for m in data if m['name'] == 'openai-gpt-5.1')

    models = [opus_46, gpt_51, sonnet_46, sonnet_45]
    names = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Sonnet 4.5']
    colors_list = [COLORS['opus_46'], COLORS['gpt_51'],
                   COLORS['sonnet_46'], COLORS['sonnet_45']]

    elos = [m['overall']['elo'] for m in models]
    win_rates = [m['overall']['win_rate'] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')

    for i, (elo, wr, name, color) in enumerate(zip(elos, win_rates, names, colors_list)):
        ax.scatter(wr, elo, s=300, color=color, alpha=0.8, edgecolor='white', linewidth=2)

        # Labels
        offset_x = 2 if i != 1 else -2
        ha = 'left' if i != 1 else 'right'
        ax.text(wr + offset_x, elo, name, ha=ha, va='center',
                fontsize=10, fontweight='500', color='#1E293B')

    ax.set_xlabel('Win Rate (%)', fontsize=12, color='#334155', fontweight='500')
    ax.set_ylabel('Overall ELO', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_xlim(35, 80)
    ax.set_ylim(1450, 1850)

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate_vs_elo.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: win_rate_vs_elo.png")
    plt.close()

def plot_sonnet_win_rate_comparison(data, output_dir):
    """Win rate comparison: Sonnet 4.6 vs 4.5"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')

    # Dataset win rates
    datasets = ['Factual QA', 'Long-form synthesis', 'Scientific verification']
    win_rates_46 = [
        sonnet_46['by_dataset']['MSMARCO']['win_rate'] * 100,
        sonnet_46['by_dataset']['PG']['win_rate'] * 100,
        sonnet_46['by_dataset']['SciFact']['win_rate'] * 100
    ]
    win_rates_45 = [
        sonnet_45['by_dataset']['MSMARCO']['win_rate'] * 100,
        sonnet_45['by_dataset']['PG']['win_rate'] * 100,
        sonnet_45['by_dataset']['SciFact']['win_rate'] * 100
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, win_rates_46, width, label='Sonnet 4.6',
                   color=COLORS['sonnet_46'], alpha=0.9,
                   edgecolor='#6B21A8', linewidth=2.5)
    bars2 = ax.bar(x + width/2, win_rates_45, width, label='Sonnet 4.5',
                   color=COLORS['sonnet_45'], alpha=0.85,
                   edgecolor='#78716C', linewidth=2.5)

    # Value labels - positioned well above bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom',
                fontweight='700', fontsize=12, color='#1E293B')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom',
                fontweight='700', fontsize=12, color='#1E293B')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13, color='#334155', fontweight='600')
    ax.set_ylabel('Win Rate (%)', fontsize=14, color='#334155', fontweight='600', labelpad=12)

    minimal_style(ax)
    ax.legend(loc='upper center', frameon=False, fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0, 85)

    plt.tight_layout()
    plt.savefig(output_dir / 'sonnet_winrate_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: sonnet_winrate_comparison.png")
    plt.close()

def plot_answer_length_comparison(output_dir):
    """Answer length comparison for Paul Graham synthesis"""
    # Token counts from actual evaluations
    models = ['Sonnet 4.5', 'Sonnet 4.6', 'GPT-5.1']
    token_counts = [336, 498, 1554]
    # Teal gradient
    colors_list = ['#06B6D4', '#14B8A6', '#10B981']  # Cyan, Teal, Emerald
    edge_colors = ['#0891B2', '#0D9488', '#059669']

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='white')

    bars = ax.bar(models, token_counts, width=0.6,
                   color=colors_list, alpha=0.9,
                   edgecolor=edge_colors, linewidth=2.5)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{height:.0f}', ha='center', va='bottom',
                fontweight='700', fontsize=12, color='#1E293B')

    ax.set_ylabel('Average Answer Length (tokens)', fontsize=14, color='#334155',
                  fontweight='600', labelpad=12)
    ax.set_xlabel('Model', fontsize=14, color='#334155', fontweight='600', labelpad=12)

    minimal_style(ax)
    ax.set_ylim(0, 1700)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=13, color='#334155', fontweight='600')

    # Add subtitle
    ax.text(0.5, 1.06, 'Essay & Narrative Content',
            transform=ax.transAxes, ha='center', fontsize=11,
            color='#64748B', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'answer_length_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: answer_length_comparison.png")
    plt.close()

def main():
    print("🎨 Generating minimalistic comparison plots...")
    print()

    data = load_data()
    output_dir = Path(__file__).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Generate all plots
    plot_overall_top5(data, output_dir)
    plot_sonnet_comparison(data, output_dir)
    plot_sonnet_win_rate_comparison(data, output_dir)
    plot_answer_length_comparison(output_dir)
    plot_sonnet_45_strength(data, output_dir)
    plot_win_rate_comparison(data, output_dir)

    print()
    print(f"✨ All plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
