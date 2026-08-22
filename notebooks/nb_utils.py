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
