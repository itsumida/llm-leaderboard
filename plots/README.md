# Visualization for Claude Opus 4.6 Blog Post

## Overview

Single high-quality plot generated with professional styling inspired by agentset.ai, Anthropic, and OpenAI design patterns.

**Color Palette:**
- **Purple (#9b59b6)**: Wins (hero color)
- **Orange (#f39c12)**: Ties (accent)
- **Gray (#ecf0f1)**: Losses (neutral)
- **Light background (#fafafa)**: Clean, modern aesthetic

---

## Generated Plot

### **dataset_breakdown.png** - Win/Loss/Tie Breakdown by Dataset
**Purpose:** Stacked bar chart showing exact win/loss/tie counts across dataset types

**Key Insights:**
- **Factual** (MSMARCO): 268-16-46 (81.2% wins) - dominant performance
- **Synthesis** (Paul Graham): 255-52-23 (77.3% wins) - strong performance
- **Scientific** (SciFact): 217-25-88 (65.8% wins) - high tie rate (26.7%)

**Usage in Blog:**
Place in dataset breakdown sections to show raw judgment distribution alongside ELO ratings and performance metrics.

**Design Features:**
- Simplified labels showing only dataset types (Factual, Synthesis, Scientific)
- No specific dataset names for cleaner presentation
- Win percentages prominently displayed in purple bars
- Stacked format shows complete picture of judgments

---

## Technical Details

**Resolution:** 300 DPI (publication quality)
**Format:** PNG with white background
**Dimensions:** 10x6 inches (optimized for web)
**Font:** Inter/Arial/Helvetica (professional sans-serif)
**Style:** Minimalist with functional clarity

**Regeneration:**
```bash
cd setup
python3 generate_plots.py
```

The plot will be regenerated in the `plots/` directory.

---

## Style Inspiration

**agentset.ai:** Gradient colors, clean borders, comparative metrics emphasis
**Anthropic:** Minimalist, high contrast, functional design
**OpenAI:** Scannable visuals, highlighting key information

The resulting style balances professional tech blog aesthetics with data clarity.
