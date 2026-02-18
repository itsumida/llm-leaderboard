#!/usr/bin/env python3
"""
Generate high-quality plots for Claude Opus 4.6 blog post.
Style inspired by agentset.ai, Anthropic, and OpenAI visualizations.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Professional color palette (inspired by agentset.ai gradient + modern tech blogs)
COLORS = {
    'opus_46': '#9b59b6',      # Purple (hero color)
    'gpt_51': '#3498db',        # Blue
    'grok_fast': '#e67e22',     # Orange
    'gemini_flash': '#e74c3c',  # Red
    'gpt_52': '#2ecc71',        # Green
    'opus_45': '#95a5a6',       # Gray
    'accent': '#f39c12',        # Gold accent
    'background': '#fafafa',    # Light background
    'text': '#2c3e50',          # Dark text
    'grid': '#ecf0f1'           # Light grid
}

# Ultra high quality font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SF Pro Display', 'Inter', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.framealpha'] = 1.0
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 150

def load_data():
    """Load benchmark data."""
    data_path = Path(__file__).parent.parent / 'final-benchmarks.json'
    with open(data_path) as f:
        return json.load(f)

def style_plot(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling - ultra clean, professional."""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e0e0e0')
    ax.spines['bottom'].set_color('#e0e0e0')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(colors=COLORS['text'], length=6, width=1.5)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=1, color='#e8e8e8', axis='x')
    ax.set_axisbelow(True)
    # No title - user requested removal
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, color=COLORS['text'], fontweight='500', labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=COLORS['text'], fontweight='500', labelpad=10)

def plot_overall_elo_rankings(data, output_dir):
    """Plot 1: Overall ELO Rankings (Top 5)."""
    # Get top 5 models by ELO
    models = sorted(data, key=lambda x: x['overall']['elo'], reverse=True)[:5]

    names = [m['name'].replace('anthropic-', '').replace('openai-', '').replace('x-ai-', '')
             .replace('google-', '').replace('claude-opus-', 'Opus ').replace('gpt-', 'GPT-')
             .replace('grok-', 'Grok ').replace('gemini-', 'Gemini ')
             .replace('-preview', '').replace('-', ' ').title()
             for m in models]
    elos = [m['overall']['elo'] for m in models]
    win_rates = [m['overall']['win_rate'] * 100 for m in models]

    # Custom colors for each model
    colors_list = [COLORS['opus_46'], COLORS['gpt_51'], COLORS['grok_fast'],
                   COLORS['gemini_flash'], COLORS['gpt_52']]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, elos, color=colors_list, alpha=0.85, height=0.6)

    # Add win rate labels on bars
    for i, (bar, elo, wr) in enumerate(zip(bars, elos, win_rates)):
        ax.text(bar.get_width() - 20, bar.get_y() + bar.get_height()/2,
                f'{elo:.0f} ELO  •  {wr:.1f}% wins',
                ha='right', va='center', color='white', fontweight='600', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    style_plot(ax, 'Overall RAG Performance Rankings', xlabel='ELO Rating')
    ax.set_xlim(1200, 1850)

    plt.tight_layout()
    plt.savefig(output_dir / 'elo_rankings.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: elo_rankings.png")
    plt.close()

def plot_dataset_performance(data, output_dir):
    """Plot 2: Performance by Dataset (Opus 4.6 vs competitors)."""
    # Focus on Opus 4.6 vs top 3 competitors
    model_names = ['anthropic-claude-opus-4-6', 'openai-gpt-5.1',
                   'x-ai-grok-4-fast', 'google-gemini-3-flash-preview']

    models = [m for m in data if m['name'] in model_names]
    models = sorted(models, key=lambda x: model_names.index(x['name']))

    datasets = ['MSMARCO', 'PG', 'SciFact']
    display_names = ['Claude Opus 4.6', 'GPT-5.1', 'Grok 4 Fast', 'Gemini 3 Flash']

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    x = np.arange(len(datasets))
    width = 0.2

    colors_list = [COLORS['opus_46'], COLORS['gpt_51'], COLORS['grok_fast'], COLORS['gemini_flash']]

    for i, (model, name, color) in enumerate(zip(models, display_names, colors_list)):
        elos = [model['by_dataset'][ds]['elo'] for ds in datasets]
        offset = width * (i - 1.5)
        ax.bar(x + offset, elos, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(['MSMARCO\n(Factual)', 'Paul Graham\n(Synthesis)', 'SciFact\n(Scientific)'])

    style_plot(ax, 'Performance Across Datasets', ylabel='ELO Rating')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False,
              framealpha=0.95, edgecolor=COLORS['grid'])
    ax.set_ylim(1000, 1950)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: dataset_performance.png")
    plt.close()

def plot_win_rates(data, output_dir):
    """Plot 3: Win Rate Distribution."""
    models = sorted(data, key=lambda x: x['overall']['win_rate'], reverse=True)[:8]

    names = [m['name'].replace('anthropic-', '').replace('openai-', '').replace('x-ai-', '')
             .replace('google-', '').replace('claude-opus-', 'Opus ').replace('gpt-', 'GPT-')
             .replace('grok-', 'Grok ').replace('gemini-', 'Gemini ').replace('deepseek-', '')
             .replace('z-ai-', '').replace('-preview', '').replace('-', ' ').title()
             for m in models]
    win_rates = [m['overall']['win_rate'] * 100 for m in models]

    # Color gradient: highlight Opus 4.6
    colors_list = [COLORS['opus_46'] if 'opus-4-6' in m['name'] else COLORS['text']
                   for m in models]
    alphas = [0.9 if 'opus-4-6' in m['name'] else 0.6 for m in models]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    y_pos = np.arange(len(names))

    # Create bars individually with different alphas
    bars = []
    for i, (y, wr, color, alpha) in enumerate(zip(y_pos, win_rates, colors_list, alphas)):
        bar = ax.barh(y, wr, color=color, alpha=alpha, height=0.65)
        bars.append(bar)

    # Add percentage labels
    for i, wr in enumerate(win_rates):
        ax.text(wr + 1, i, f'{wr:.1f}%', ha='left', va='center',
                color=COLORS['text'], fontweight='500', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()

    style_plot(ax, 'Win Rate Against All Competitors', xlabel='Win Rate (%)')
    ax.set_xlim(0, 85)

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rates.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: win_rates.png")
    plt.close()

def plot_latency_vs_quality(data, output_dir):
    """Plot 4: Latency vs Quality Trade-off."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

    # Extract data
    latencies = [m['overall']['avg_latency_ms'] / 1000 for m in data]  # Convert to seconds
    quality = [m['overall']['avg_metric_overall'] for m in data]
    names = [m['name'].replace('anthropic-', '').replace('openai-', '').replace('x-ai-', '')
             .replace('google-', '').replace('claude-opus-', 'Opus ').replace('gpt-', 'GPT-')
             .replace('grok-', 'Grok ').replace('gemini-', 'Gemini ').replace('deepseek-', '')
             .replace('z-ai-', '').replace('qwen-', '').replace('-preview', '').replace('-', ' ')
             .replace('qwen3 30b a3b thinking 2507', 'Qwen3').title()
             for m in data]

    # Color by model
    colors_list = []
    sizes = []
    for m in data:
        if 'opus-4-6' in m['name']:
            colors_list.append(COLORS['opus_46'])
            sizes.append(250)
        elif 'gpt-5.1' in m['name']:
            colors_list.append(COLORS['gpt_51'])
            sizes.append(200)
        elif 'grok' in m['name']:
            colors_list.append(COLORS['grok_fast'])
            sizes.append(200)
        else:
            colors_list.append(COLORS['text'])
            sizes.append(120)

    # Scatter plot
    scatter = ax.scatter(latencies, quality, c=colors_list, s=sizes, alpha=0.7,
                        edgecolors='white', linewidth=2)

    # Annotate key models
    annotate_models = ['Opus 4 6', 'Gpt 5 1', 'Grok 4 Fast']
    for i, name in enumerate(names):
        if any(am.lower() in name.lower() for am in annotate_models):
            ax.annotate(name, (latencies[i], quality[i]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='600', color=COLORS['text'],
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=COLORS['grid'], alpha=0.9))

    style_plot(ax, 'Latency vs Quality Trade-off',
               xlabel='Average Latency (seconds)', ylabel='Average Quality Score')
    ax.set_ylim(4.7, 5.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_quality.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: latency_quality.png")
    plt.close()

def plot_quality_metrics_radar(data, output_dir):
    """Plot 5: Quality Metrics Radar Chart (Opus 4.6 SciFact)."""
    # Get Opus 4.6 and GPT-5.1 SciFact metrics
    opus_46 = next(m for m in data if 'opus-4-6' in m['name'])
    gpt_51 = next(m for m in data if 'gpt-5.1' in m['name'])

    metrics = ['Correctness', 'Faithfulness', 'Grounding', 'Relevance', 'Completeness']
    opus_values = [opus_46['by_dataset']['SciFact']['metrics'][m.lower()] for m in metrics]
    gpt_values = [gpt_51['by_dataset']['SciFact']['metrics'][m.lower()] for m in metrics]

    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the plot
    opus_values += opus_values[:1]
    gpt_values += gpt_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), facecolor='white')

    # Plot data
    ax.plot(angles, opus_values, 'o-', linewidth=2, label='Claude Opus 4.6',
            color=COLORS['opus_46'], markersize=8)
    ax.fill(angles, opus_values, alpha=0.25, color=COLORS['opus_46'])

    ax.plot(angles, gpt_values, 'o-', linewidth=2, label='GPT-5.1',
            color=COLORS['gpt_51'], markersize=8)
    ax.fill(angles, gpt_values, alpha=0.15, color=COLORS['gpt_51'])

    # Fix axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, color=COLORS['text'])
    ax.set_ylim(4, 5.2)
    ax.set_yticks([4.2, 4.4, 4.6, 4.8, 5.0])
    ax.set_yticklabels(['4.2', '4.4', '4.6', '4.8', '5.0'], fontsize=9, color=COLORS['text'])
    ax.grid(True, alpha=0.3)

    ax.set_title('Quality Metrics: SciFact Dataset\nHighlighting Completeness Gap',
                 fontsize=14, fontweight='600', color=COLORS['text'], pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
              framealpha=0.95, edgecolor=COLORS['grid'])

    plt.tight_layout()
    plt.savefig(output_dir / 'quality_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: quality_radar.png")
    plt.close()

def plot_head_to_head_matrix(data, output_dir):
    """Plot 6: Head-to-Head Win Rate Matrix (Opus 4.6 vs all)."""
    opus_46 = next(m for m in data if 'opus-4-6' in m['name'])

    # Get all comparisons
    opponents = list(opus_46['comparisons'].keys())
    win_rates = [opus_46['comparisons'][opp]['win_rate'] * 100 for opp in opponents]

    # Sort by win rate
    sorted_pairs = sorted(zip(opponents, win_rates), key=lambda x: x[1], reverse=True)
    opponents, win_rates = zip(*sorted_pairs)

    # Clean names
    names = [opp.replace('anthropic-', '').replace('openai-', '').replace('x-ai-', '')
             .replace('google-', '').replace('claude-opus-', 'Opus ').replace('gpt-', 'GPT-')
             .replace('grok-', 'Grok ').replace('gemini-', 'Gemini ').replace('deepseek-', '')
             .replace('z-ai-', '').replace('qwen-', '').replace('-preview', '').replace('-', ' ')
             .replace('qwen3 30b a3b thinking 2507', 'Qwen3').replace('Gpt Oss 120b', 'GPT-OSS').title()
             for opp in opponents]

    # Color gradient based on win rate
    colors_list = []
    for wr in win_rates:
        if wr >= 80:
            colors_list.append(COLORS['opus_46'])
        elif wr >= 60:
            colors_list.append(COLORS['gpt_51'])
        elif wr >= 40:
            colors_list.append(COLORS['accent'])
        else:
            colors_list.append(COLORS['gemini_flash'])

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, win_rates, color=colors_list, alpha=0.8, height=0.7)

    # Add win rate labels
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f'{wr:.1f}%', ha='left', va='center', color=COLORS['text'],
                fontweight='500', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()

    # Add vertical line at 50%
    ax.axvline(x=50, color=COLORS['text'], linestyle='--', linewidth=1, alpha=0.3)
    ax.text(50, -0.8, '50%', ha='center', va='top', color=COLORS['text'],
            fontsize=9, fontweight='500')

    style_plot(ax, 'Claude Opus 4.6: Head-to-Head Win Rates', xlabel='Win Rate vs Opponent (%)')
    ax.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'head_to_head.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: head_to_head.png")
    plt.close()

def plot_dataset_wins_breakdown(data, output_dir):
    """Plot 7: Win/Loss/Tie Breakdown by Dataset."""
    opus_46 = next(m for m in data if 'opus-4-6' in m['name'])

    datasets = ['MSMARCO', 'PG', 'SciFact']
    wins = [opus_46['by_dataset'][ds]['wins'] for ds in datasets]
    losses = [opus_46['by_dataset'][ds]['losses'] for ds in datasets]
    ties = [opus_46['by_dataset'][ds]['ties'] for ds in datasets]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    x = np.arange(len(datasets))
    width = 0.6

    # Stacked bar chart
    p1 = ax.bar(x, wins, width, label='Wins', color=COLORS['opus_46'], alpha=0.9)
    p2 = ax.bar(x, ties, width, bottom=wins, label='Ties', color=COLORS['accent'], alpha=0.7)
    p3 = ax.bar(x, losses, width, bottom=np.array(wins)+np.array(ties),
                label='Losses', color=COLORS['grid'], alpha=0.7)

    # Add percentage labels
    totals = np.array(wins) + np.array(losses) + np.array(ties)
    for i, (w, t, l, total) in enumerate(zip(wins, ties, losses, totals)):
        win_pct = (w / total) * 100
        # Label on wins section
        ax.text(i, w/2, f'{w}\n({win_pct:.1f}%)', ha='center', va='center',
                color='white', fontweight='600', fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(['Factual', 'Synthesis', 'Scientific'])

    style_plot(ax, ylabel='Number of Judgments')
    ax.legend(loc='upper right', frameon=True, framealpha=1.0, edgecolor='#e0e0e0', shadow=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_breakdown.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: dataset_breakdown.png")
    plt.close()

def main():
    """Generate all plots."""
    print("🎨 Generating high-quality plots for Claude Opus 4.6 blog post...")
    print()

    # Load data
    data = load_data()

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    # plot_overall_elo_rankings(data, output_dir)
    # plot_dataset_performance(data, output_dir)
    # plot_win_rates(data, output_dir)
    # plot_latency_vs_quality(data, output_dir)
    # plot_quality_metrics_radar(data, output_dir)
    # plot_head_to_head_matrix(data, output_dir)
    plot_dataset_wins_breakdown(data, output_dir)

    print()
    print(f"✨ Plot saved to: {output_dir}")
    print()
    print("Generated plot:")
    print("  dataset_breakdown.png - Win/loss/tie breakdown by dataset")

if __name__ == '__main__':
    main()
