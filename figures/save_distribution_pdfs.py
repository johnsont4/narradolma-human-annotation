"""
Save per-feature annotation distribution PDFs.

Reads the exported parquets in ../data/. ANNOTATOR selects which label set to
plot: 'gold' for the adjudicated labels used in the paper, or any annotator
handle (e.g. 'maria') for that individual's distributions.

Output directory: ./pdfs/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ANNOTATOR   = 'gold'
_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, '..', 'data')
OUT_DIR     = os.path.join(_HERE, 'pdfs')
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_SETTING = '#D4B896'
COLOR_AGENCY  = '#A8B78B'
COLOR_EVENT   = '#7BA8A8'

# ── load data ──────────────────────────────────────────────────────────────────
setting_df = pd.read_parquet(os.path.join(DATA_DIR, 'setting_annotations.parquet'))
agency_df  = pd.read_parquet(os.path.join(DATA_DIR, 'agency_annotations.parquet'))
event_df   = pd.read_parquet(os.path.join(DATA_DIR, 'event_relation_annotations.parquet'))

setting_cols = [c for c in setting_df.columns if c.endswith(f'_{ANNOTATOR}')]
agency_cols  = [c for c in agency_df.columns  if c.endswith(f'_{ANNOTATOR}')]
event_cols   = [c for c in event_df.columns   if c.endswith(f'_{ANNOTATOR}')]

SCALE_15 = [1, 2, 3, 4, 5]


def short(col):
    return (col.replace(f'_{ANNOTATOR}', '')
               .replace('agency_', '')
               .replace('setting_', '')
               .replace('_', ' ')
               .title())


def save_likert(df, col, scale, color, prefix):
    counts = pd.to_numeric(df[col], errors='coerce').value_counts().reindex(scale, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=color, edgecolor='white', width=0.6)
    ax.set_xticks(scale)
    ax.set_xlabel('Rating (1–5)')
    ax.set_ylabel('Count')
    ax.set_title(short(col))
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'{prefix}_{col.replace(f"_{ANNOTATOR}", "")}.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f'saved {fname}')


def save_categorical(df, col, color, prefix):
    counts = df[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(4, len(counts) * 1.2), 4))
    ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor='white', width=0.6)
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title(short(col))
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'{prefix}_{col.replace(f"_{ANNOTATOR}", "")}.pdf')
    fig.savefig(fname)
    plt.close(fig)
    print(f'saved {fname}')


def save_binary(series, title, labels, color, filename):
    counts = series.value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(counts.index.astype(str), counts.values, color=color, edgecolor='white', width=0.6)
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title(title)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, filename)
    fig.savefig(fname)
    plt.close(fig)
    print(f'saved {fname}')


# ── setting (1–5 Likert) ───────────────────────────────────────────────────────
for col in setting_cols:
    save_likert(setting_df, col, SCALE_15, COLOR_SETTING, 'setting')

# ── agency (1–5 Likert) ────────────────────────────────────────────────────────
for col in agency_cols:
    save_likert(agency_df, col, SCALE_15, COLOR_AGENCY, 'agency')

# ── event relation (categorical) ───────────────────────────────────────────────
for col in event_cols:
    save_categorical(event_df, col, COLOR_EVENT, 'event')

# ── binary: causal vs. not causal ─────────────────────────────────────────────
causality_col = f'causality_rating_{ANNOTATOR}'
causal_binary = event_df[causality_col].map(
    lambda v: 'causal' if v in ('direct_cause', 'enables') else 'not causal'
)
save_binary(causal_binary, 'Causal vs. Not Causal',
            ['causal', 'not causal'], COLOR_EVENT,
            'event_causality_binary.pdf')

# ── binary: temporal vs. not temporal ─────────────────────────────────────────
temporal_col = f'temporal_order_{ANNOTATOR}'
NON_TEMPORAL = {'too_hard_to_tell'}
temporal_binary = event_df[temporal_col].map(
    lambda v: 'not temporal' if v in NON_TEMPORAL else 'temporal'
)
save_binary(temporal_binary, 'Temporal vs. Not Temporal',
            ['temporal', 'not temporal'], COLOR_EVENT,
            'event_temporal_binary.pdf')

print(f'\nAll PDFs written to {OUT_DIR}')
