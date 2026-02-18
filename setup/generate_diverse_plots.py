#!/usr/bin/env python3
"""
Generate diverse plot types for Sonnet comparison blog post
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

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

def plot_radar_quality_metrics(data, output_dir):
    """Radar chart comparing quality metrics"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')
    opus_46 = next(m for m in data if m['name'] == 'anthropic-claude-opus-4-6')
    gpt_51 = next(m for m in data if m['name'] == 'openai-gpt-5.1')

    # Metrics
    categories = ['Correctness', 'Faithfulness', 'Grounding', 'Relevance', 'Completeness']

    # Get average metrics across all datasets
    def get_avg_metrics(model):
        all_metrics = []
        for dataset in ['MSMARCO', 'PG', 'SciFact']:
            m = model['by_dataset'][dataset]['metrics']
            all_metrics.append([
                m['correctness'],
                m['faithfulness'],
                m['grounding'],
                m['relevance'],
                m['completeness']
            ])
        return np.mean(all_metrics, axis=0)

    values_46 = get_avg_metrics(sonnet_46)
    values_45 = get_avg_metrics(sonnet_45)
    values_opus = get_avg_metrics(opus_46)
    values_gpt = get_avg_metrics(gpt_51)

    # Number of variables
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    values_46 = np.concatenate((values_46, [values_46[0]]))
    values_45 = np.concatenate((values_45, [values_45[0]]))
    values_opus = np.concatenate((values_opus, [values_opus[0]]))
    values_gpt = np.concatenate((values_gpt, [values_gpt[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), facecolor='white')

    # Plot data
    ax.plot(angles, values_opus, 'o-', linewidth=2, label='Opus 4.6',
            color=COLORS['opus_46'], alpha=0.8)
    ax.fill(angles, values_opus, alpha=0.15, color=COLORS['opus_46'])

    ax.plot(angles, values_gpt, 'o-', linewidth=2, label='GPT-5.1',
            color=COLORS['gpt_51'], alpha=0.8)
    ax.fill(angles, values_gpt, alpha=0.15, color=COLORS['gpt_51'])

    ax.plot(angles, values_46, 'o-', linewidth=2.5, label='Sonnet 4.6',
            color=COLORS['sonnet_46'], alpha=0.9)
    ax.fill(angles, values_46, alpha=0.2, color=COLORS['sonnet_46'])

    ax.plot(angles, values_45, 'o--', linewidth=2, label='Sonnet 4.5',
            color=COLORS['sonnet_45'], alpha=0.8)
    ax.fill(angles, values_45, alpha=0.12, color=COLORS['sonnet_45'])

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, color='#334155')

    # Set y-axis limits
    ax.set_ylim(4.0, 5.0)
    ax.set_yticks([4.2, 4.4, 4.6, 4.8, 5.0])
    ax.set_yticklabels(['4.2', '4.4', '4.6', '4.8', '5.0'], size=9, color='#64748B')

    # Grid styling
    ax.grid(color=COLORS['grid'], linestyle='-', linewidth=1, alpha=0.3)
    ax.spines['polar'].set_visible(False)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'quality_radar.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: quality_radar.png")
    plt.close()

def plot_heatmap_dataset_performance(data, output_dir):
    """Heatmap showing model performance across datasets"""
    models_to_show = [
        'anthropic-claude-opus-4-6',
        'openai-gpt-5.1',
        'anthropic-claude-sonnet-4.6',
        'x-ai-grok-4-fast',
        'google-gemini-3-flash-preview',
        'anthropic-claude-opus-4.5',
        'anthropic-claude-sonnet-4.5'
    ]

    models = [m for m in data if m['name'] in models_to_show]
    models = sorted(models, key=lambda x: x['overall']['elo'], reverse=True)

    # Get win rates for each dataset
    datasets = ['MSMARCO', 'PG', 'SciFact']
    dataset_labels = ['Factual', 'Synthesis', 'Scientific']
    model_labels = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Grok 4 Fast',
                    'Gemini 3 Flash', 'Opus 4.5', 'Sonnet 4.5']

    heatmap_data = []
    for model in models:
        row = [model['by_dataset'][ds]['win_rate'] * 100 for ds in datasets]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor='white')

    # Normalize data for color mapping (0-100% range)
    norm_data = heatmap_data / 100

    # Professional teal colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#F0FDFA', '#CCFBF1', '#5EEAD4', '#14B8A6', '#0D9488']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)

    # Create heatmap
    im = ax.imshow(norm_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Add grid lines (borders between cells)
    for i in range(len(model_labels) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(len(datasets) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=2)

    # Set ticks
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(dataset_labels, fontsize=11, color='#334155', fontweight='600')
    ax.set_yticklabels(model_labels, fontsize=11, color='#334155', fontweight='500')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add text annotations
    for i in range(len(model_labels)):
        for j in range(len(datasets)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}%',
                          ha="center", va="center",
                          color='white' if norm_data[i, j] > 0.6 else '#1E293B',
                          fontsize=11, fontweight='700')

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Win Rate (%)', rotation=270, labelpad=20, fontsize=11, color='#334155')
    cbar.ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color='#64748B')
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_heatmap.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: dataset_heatmap.png")
    plt.close()

def plot_line_elo_progression(data, output_dir):
    """Line chart showing ELO across datasets"""
    sonnet_46 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.6')
    sonnet_45 = next(m for m in data if m['name'] == 'anthropic-claude-sonnet-4.5')
    opus_46 = next(m for m in data if m['name'] == 'anthropic-claude-opus-4-6')
    gpt_51 = next(m for m in data if m['name'] == 'openai-gpt-5.1')

    datasets = ['MS MARCO', 'Paul Graham', 'SciFact']
    dataset_keys = ['MSMARCO', 'PG', 'SciFact']

    models = [opus_46, gpt_51, sonnet_46, sonnet_45]
    names = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Sonnet 4.5']
    colors_list = [COLORS['opus_46'], COLORS['gpt_51'],
                   COLORS['sonnet_46'], COLORS['sonnet_45']]
    styles = ['-', '-', '-', '--']
    widths = [2.5, 2.5, 3, 2]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    for model, name, color, style, width in zip(models, names, colors_list, styles, widths):
        elos = [model['by_dataset'][k]['elo'] for k in dataset_keys]
        ax.plot(datasets, elos, marker='o', linewidth=width, linestyle=style,
                label=name, color=color, alpha=0.9, markersize=8)

    # Styling
    ax.set_xlabel('Dataset Type', fontsize=12, color='#334155', fontweight='500')
    ax.set_ylabel('ELO Rating', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_ylim(1200, 1950)
    ax.legend(loc='best', frameon=False, fontsize=11)

    # Add markers for Sonnet 4.5's SciFact dominance
    s45_scifact_elo = sonnet_45['by_dataset']['SciFact']['elo']
    ax.annotate('Sonnet 4.5\ndominates!',
                xy=(2, s45_scifact_elo), xytext=(1.5, s45_scifact_elo + 100),
                fontsize=9, color=COLORS['sonnet_45'], fontweight='600',
                arrowprops=dict(arrowstyle='->', color=COLORS['sonnet_45'],
                               lw=1.5, alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'elo_line_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: elo_line_chart.png")
    plt.close()

def plot_stacked_area_win_distribution(data, output_dir):
    """Stacked area chart showing win/loss/tie distribution"""
    models_to_show = [
        'anthropic-claude-opus-4-6',
        'openai-gpt-5.1',
        'anthropic-claude-sonnet-4.6',
        'anthropic-claude-sonnet-4.5'
    ]

    models = [m for m in data if m['name'] in models_to_show]
    models = sorted(models, key=lambda x: x['overall']['elo'], reverse=True)

    names = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Sonnet 4.5']

    # Calculate win/tie/loss percentages
    wins = []
    ties = []
    losses = []

    for model in models:
        overall = model['overall']
        total = overall['wins'] + overall['ties'] + overall['losses']
        wins.append(overall['wins'] / total * 100)
        ties.append(overall['ties'] / total * 100)
        losses.append(overall['losses'] / total * 100)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    x = np.arange(len(names))

    # Create stacked bars
    p1 = ax.bar(x, wins, label='Wins', color='#10B981', alpha=0.9)
    p2 = ax.bar(x, ties, bottom=wins, label='Ties', color='#F59E0B', alpha=0.8)
    p3 = ax.bar(x, losses, bottom=np.array(wins) + np.array(ties),
                label='Losses', color='#EF4444', alpha=0.8)

    # Add percentage labels
    for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
        # Win label
        if w > 8:
            ax.text(i, w/2, f'{w:.0f}%', ha='center', va='center',
                   color='white', fontweight='600', fontsize=10)
        # Tie label
        if t > 5:
            ax.text(i, w + t/2, f'{t:.0f}%', ha='center', va='center',
                   color='white', fontweight='600', fontsize=9)
        # Loss label
        if l > 8:
            ax.text(i, w + t + l/2, f'{l:.0f}%', ha='center', va='center',
                   color='white', fontweight='600', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, color='#334155')
    ax.set_ylabel('Distribution (%)', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'win_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: win_distribution.png")
    plt.close()

def plot_bubble_cost_performance(data, output_dir):
    """Bubble chart: ELO vs Win Rate, bubble size = cost efficiency"""
    models_to_show = [
        'anthropic-claude-opus-4-6',
        'openai-gpt-5.1',
        'anthropic-claude-sonnet-4.6',
        'x-ai-grok-4-fast',
        'google-gemini-3-flash-preview',
        'anthropic-claude-sonnet-4.5'
    ]

    models = [m for m in data if m['name'] in models_to_show]

    # Model costs (input/output per 1M tokens at 10:1 ratio)
    costs = {
        'anthropic-claude-opus-4-6': 7.50,
        'openai-gpt-5.1': 2.13,
        'anthropic-claude-sonnet-4.6': 4.50,
        'x-ai-grok-4-fast': 0.25,
        'google-gemini-3-flash-preview': 0.80,
        'anthropic-claude-sonnet-4.5': 1.50
    }

    names = ['Opus 4.6', 'GPT-5.1', 'Sonnet 4.6', 'Grok 4 Fast',
             'Gemini 3 Flash', 'Sonnet 4.5']
    colors_list = [COLORS['opus_46'], COLORS['gpt_51'], COLORS['sonnet_46'],
                   '#10B981', '#EF4444', COLORS['sonnet_45']]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

    for model, name, color in zip(models, names, colors_list):
        elo = model['overall']['elo']
        win_rate = model['overall']['win_rate'] * 100
        cost = costs[model['name']]

        # Bubble size inversely proportional to cost (efficiency = elo/cost)
        efficiency = elo / cost
        size = efficiency / 2  # Scale down for visibility

        ax.scatter(win_rate, elo, s=size, color=color, alpha=0.7,
                  edgecolor='white', linewidth=2)

        # Label
        offset_x = 2 if name != 'GPT-5.1' else -2
        offset_y = 20 if name not in ['Grok 4 Fast', 'Gemini 3 Flash'] else -30
        ha = 'left' if name != 'GPT-5.1' else 'right'

        ax.annotate(f'{name}\n${cost:.2f}/1M',
                   xy=(win_rate, elo),
                   xytext=(win_rate + offset_x, elo + offset_y),
                   fontsize=9, color='#1E293B', fontweight='500',
                   ha=ha, va='center')

    ax.set_xlabel('Win Rate (%)', fontsize=12, color='#334155', fontweight='500')
    ax.set_ylabel('Overall ELO', fontsize=12, color='#334155', fontweight='500')

    minimal_style(ax)
    ax.set_xlim(35, 80)
    ax.set_ylim(1450, 1850)

    # Add legend for bubble size
    legend_text = 'Bubble size = ELO / Cost\n(larger = better value)'
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
           fontsize=9, color='#64748B', ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E2E8F0', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / 'cost_efficiency_bubble.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: cost_efficiency_bubble.png")
    plt.close()

def main():
    print("🎨 Generating diverse plot types...")
    print()

    data = load_data()
    output_dir = Path(__file__).parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Generate all diverse plots
    plot_radar_quality_metrics(data, output_dir)
    plot_heatmap_dataset_performance(data, output_dir)
    plot_line_elo_progression(data, output_dir)
    plot_stacked_area_win_distribution(data, output_dir)
    plot_bubble_cost_performance(data, output_dir)

    print()
    print(f"✨ All diverse plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
