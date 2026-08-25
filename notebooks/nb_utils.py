"""
nb_utils.py

Data loading for the analysis notebooks. Every notebook reads parquet from
data/ and nothing else -- there is no path back into the potato repo.

    from nb_utils import load, DIMENSIONS, annotators_for

    df = load('all')          # or 'setting' / 'agency' / 'event_relation' / 'corpus'

To refresh the parquets after a new round of annotation, run
`python export_annotations.py` in potato/teagan_tasks.
"""

import os
import re

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

_FILES = {
    'setting':        'setting_annotations.parquet',
    'agency':         'agency_annotations.parquet',
    'event_relation': 'event_relation_annotations.parquet',
    'all':            'all_annotations.parquet',
    'corpus':         'corpus.parquet',
}

# Label dimensions per task, in the order they are presented in the UI.
DIMENSIONS = {
    'setting': [
        'setting_concreteness',
        'setting_temporal_grounding',
        'setting_spatial_grounding',
        'setting_sensory',
    ],
    'agency': [
        'agency_focalization',
        'agency_emotion',
        'agency_cognition',
        'agency_change_of_state',
        'agency_conflict',
    ],
    'event_relation': [
        'span1_is_event',
        'span2_is_event',
        'temporal_order',
        'causality_rating',
    ],
}

# The 1-5 Likert dimensions (event_relation uses its own scales).
LIKERT_TASKS = ('setting', 'agency')
SCALE_15 = [1, 2, 3, 4, 5]


def load(name='all'):
    """Read one of the exported parquets. See _FILES for valid names."""
    if name not in _FILES:
        raise KeyError(f'unknown dataset {name!r}; expected one of {sorted(_FILES)}')
    path = os.path.join(DATA_DIR, _FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Refresh it by running export_annotations.py '
            'in potato/teagan_tasks.'
        )
    return pd.read_parquet(path)


def annotators_for(df, dimension):
    """Annotator suffixes present for a dimension, e.g. ['gold', 'maria', ...].

    Reads the columns actually in the frame rather than a hardcoded list, so a
    newly added annotator appears without editing anything here.
    """
    pattern = re.compile(rf'^{re.escape(dimension)}_(.+)$')
    return [m.group(1) for c in df.columns if (m := pattern.match(c))]


def label_columns(df, task, suffix='gold'):
    """The {dim}_{suffix} columns for a task that exist in df."""
    return [f'{d}_{suffix}' for d in DIMENSIONS[task]
            if f'{d}_{suffix}' in df.columns]


# -- Coverage: who annotated which corpus indexes ------------------------------
#
# Every task is served from the same 1072-row queue (corpus.parquet, same order
# as the CSV potato serves), and the trailing integer of safe_instance_id is
# that row's position in it. An annotator did an item iff their
# annotation_order_{annotator} is non-null, so coverage is readable straight off
# the task parquets -- which are outer-joined over annotators, and therefore
# hold a row for anything anyone touched, not just what was adjudicated.

TASKS = ('setting', 'agency', 'event_relation')

_ORDER_RE = re.compile(r'^annotation_order_(.+)$')


def instance_index(ids):
    """Corpus row index encoded in each id: 'inst_<hash>_<idx>' -> idx."""
    return pd.Series(ids).str.rsplit('_', n=1).str[-1].astype(int)


def order_annotators(df):
    """Annotator suffixes on the annotation_order_* columns present in df."""
    return [m.group(1) for c in df.columns if (m := _ORDER_RE.match(c))]


def annotation_index(task=None):
    """One row per (task, annotator, item that annotator actually annotated).

    Columns: task, annotator, idx, safe_instance_id, annotation_order,
    n_dims_answered. This is the backbone the other coverage helpers derive
    from. Reads the per-task parquets, never 'all' -- the outer join there
    collides the annotation_order_* columns into _x/_y suffixes.
    """
    tasks = TASKS if task is None else (task,)
    frames = []
    for t in tasks:
        df = load(t)
        idx = instance_index(df['safe_instance_id'])
        for ann in order_annotators(df):
            done = df[f'annotation_order_{ann}'].notna()
            dim_cols = [f'{d}_{ann}' for d in DIMENSIONS[t]
                        if f'{d}_{ann}' in df.columns]
            frames.append(pd.DataFrame({
                'task': t,
                'annotator': ann,
                'idx': idx[done].to_numpy(),
                'safe_instance_id': df.loc[done, 'safe_instance_id'].to_numpy(),
                'annotation_order': df.loc[done, f'annotation_order_{ann}']
                                      .astype(int).to_numpy(),
                'n_dims_answered': df.loc[done, dim_cols].notna()
                                     .sum(axis=1).to_numpy(),
            }))
    out = pd.concat(frames, ignore_index=True)
    out['task'] = pd.Categorical(out['task'], categories=TASKS, ordered=True)
    return out.sort_values(['task', 'annotator', 'idx'], ignore_index=True)


def coverage(task=None):
    """Per (task, annotator): how many items, which index block, what was skipped.

    n_skipped_in_range counts positions inside [idx_min, idx_max] the annotator
    passed over; open items beyond idx_max are not skips, they are unreached.
    n_complete_all_dims counts items where every dimension of the task was
    answered -- event_relation has many partial rows.
    """
    ann_idx = annotation_index(task)
    records = []
    for (t, ann), g in ann_idx.groupby(['task', 'annotator'], observed=True):
        lo, hi = int(g['idx'].min()), int(g['idx'].max())
        span = hi - lo + 1
        records.append({
            'task': t,
            'annotator': ann,
            'n_items': len(g),
            'idx_min': lo,
            'idx_max': hi,
            'span': span,
            'n_skipped_in_range': span - len(g),
            'contiguous': span == len(g),
            'n_complete_all_dims': int((g['n_dims_answered'] == len(DIMENSIONS[t])).sum()),
        })
    out = pd.DataFrame(records)
    out['task'] = pd.Categorical(out['task'], categories=TASKS, ordered=True)
    return out.sort_values(['task', 'n_items'], ascending=[True, False],
                           ignore_index=True)


def overlap(task):
    """Square matrix of shared items between annotators on a task.

    Off-diagonal is how many indexes the pair both annotated; the diagonal is
    that annotator's own total. Rows are ordered largest contributor first.
    """
    ann_idx = annotation_index(task)
    sets = {ann: set(g['idx']) for ann, g in ann_idx.groupby('annotator')}
    names = sorted(sets, key=lambda a: -len(sets[a]))
    return pd.DataFrame(
        [[len(sets[a] & sets[b]) for b in names] for a in names],
        index=names, columns=names,
    )


def items_by_annotator_count(task):
    """How many items on a task carry 1, 2, 3, ... annotators."""
    ann_idx = annotation_index(task)
    per_item = ann_idx.groupby('idx')['annotator'].nunique()
    counts = per_item.value_counts().sort_index()
    counts.index.name = 'n_annotators'
    return counts.rename('n_items')


def open_indexes(task=None):
    """Corpus indexes nobody has annotated. With task=None, untouched by any task."""
    universe = set(range(len(load('corpus'))))
    return sorted(universe - set(annotation_index(task)['idx']))


def as_ranges(indexes):
    """Collapse a sorted index list into (start, end) runs: [0,1,2,7] -> [(0,2),(7,7)]."""
    runs = []
    for i in sorted(indexes):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def format_ranges(indexes, max_runs=8):
    """as_ranges rendered for printing: '0-44, 70-98, 400-1071'.

    When there are more runs than fit, the last one is kept -- for an
    annotation queue the unreached tail is the run worth seeing.
    """
    runs = as_ranges(indexes)
    fmt = lambda r: f'{r[0]}-{r[1]}' if r[0] != r[1] else str(r[0])
    if len(runs) <= max_runs:
        return ', '.join(fmt(r) for r in runs) or '(none)'
    head = runs[:max_runs - 1]
    return (', '.join(fmt(r) for r in head)
            + f', ... (+{len(runs) - max_runs} more), ' + fmt(runs[-1]))


# -- Shared plotting setup -----------------------------------------------------
#
# Every notebook used to repeat the same seaborn/matplotlib preamble, and the
# coverage notebook had a second, different one. Both live here now so the
# figures across the repo look like one set.

# Categorical hues in fixed order: a series keeps its color wherever it appears,
# and the order is a CVD-safe sequence validated on adjacent pairs.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
INK_MUTED = '#8a8981'
GRID      = '#e5e4e0'

# Per-task accent colors, used where a chart is about one task rather than
# several series.
TASK_COLOR = {
    'setting':        '#4caf50',
    'agency':         '#f4a432',
    'event_relation': '#5b8db8',
}


def setup_plots(minimal=False):
    """Apply the shared figure style. Call once, near the top of a notebook.

    minimal=True drops the spines and grid to near-nothing, for charts whose
    marks carry the structure themselves (the coverage lanes, for instance).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style='whitegrid', font_scale=1.05)
    mpl.rcParams['figure.dpi'] = 120

    if minimal:
        mpl.rcParams.update({
            'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
            'savefig.facecolor': SURFACE,
            'text.color': INK, 'axes.labelcolor': INK_2,
            'xtick.color': INK_2, 'ytick.color': INK_2,
            'axes.edgecolor': GRID, 'grid.color': GRID, 'grid.linewidth': 0.8,
            'axes.spines.top': False, 'axes.spines.right': False,
            'axes.spines.left': False,
            'font.size': 10, 'axes.titlesize': 12, 'figure.dpi': 110,
        })
    return plt


def annotator_labels(task, annotator, dims=None, dropna=True):
    """safe_instance_id + one column per dimension, for a single annotator.

    Columns are renamed to the dimension name as declared in DIMENSIONS, so
    downstream code reads `setting_sensory` rather than
    `setting_sensory_roda9210`. Replaces the three near-identical helpers the
    notebooks each used to define.
    """
    df   = load(task)
    dims = dims if dims is not None else DIMENSIONS[task]
    cols = {f'{d}_{annotator}': d for d in dims if f'{d}_{annotator}' in df.columns}
    out  = df[['safe_instance_id'] + list(cols)].rename(columns=cols)
    return out.dropna(subset=list(cols.values()), how='all') if dropna else out


def short_name(dim, task=None):
    """'setting_temporal_grounding' -> 'Temporal Grounding'."""
    if task:
        dim = dim.replace(f'{task}_', '')
    else:
        for t in TASKS:
            dim = dim.replace(f'{t}_', '')
    return dim.replace('_', ' ').title()
