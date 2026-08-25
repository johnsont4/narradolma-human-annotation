"""
agreement.py

Inter-annotator agreement metrics and pair collection, extracted from the
agreement notebook so the metrics are importable, reusable and testable rather
than re-declared inline.

    from agreement import load_annotations, agreement_rows, TASK_LABEL

    all_data, annotators = load_annotations()
    rows = agreement_rows('setting', all_data, annotators)

`all_data` is task -> annotator -> {safe_instance_id: {dimension: value}}.
The adjudicated 'gold' copy is excluded, since it duplicates whichever
annotator did the adjudication.
"""

import numpy as np
import pandas as pd
import krippendorff

from nb_utils import load, DIMENSIONS, TASKS, annotators_for

TASK_LABEL = {
    'setting':        'Setting',
    'agency':         'Agency',
    'event_relation': 'Event Relation',
}

# Likert tasks are scored as ordinal; event_relation is nominal/binary throughout.
TASK_FORMAT = {
    'setting':        'likert',
    'agency':         'likert',
    'event_relation': 'event_relation',
}

MIN_PAIRS = 3   # fewest shared instances worth reporting a metric for


# -- Metrics -------------------------------------------------------------------
# Each takes a list of (value_a, value_b) pairs and returns None when the input
# is too small or degenerate, so callers can render an em dash.

def exact_match(pairs):
    if not pairs: return None
    return sum(a == b for a, b in pairs) / len(pairs)


def within_one(pairs):
    if not pairs: return None
    return sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)


def mae(pairs):
    if not pairs: return None
    return float(np.mean([abs(a - b) for a, b in pairs]))


def cohen_kappa(pairs):
    """Cohen's κ for nominal/binary data with 2 raters."""
    if len(pairs) < 2: return None
    n = len(pairs)
    y1, y2 = zip(*pairs)
    cats = sorted(set(y1) | set(y2), key=str)
    if len(cats) < 2: return None
    p_o = sum(a == b for a, b in pairs) / n
    p_e = sum((y1.count(c) / n) * (y2.count(c) / n) for c in cats)
    if p_e == 1: return None
    return (p_o - p_e) / (1 - p_e)


def alpha(pairs, binary=False):
    """Krippendorff's alpha: nominal for binary/categorical, interval for ordinal."""
    if len(pairs) < 2: return None
    y1, y2 = zip(*pairs)
    all_vals = sorted(set(y1) | set(y2), key=str)
    if len(all_vals) < 2: return None
    if not all(isinstance(v, (int, float)) for v in all_vals):
        enc = {v: i for i, v in enumerate(all_vals)}
        y1 = [enc[v] for v in y1]
        y2 = [enc[v] for v in y2]
    level = 'nominal' if binary else 'interval'
    data = np.array([list(y1), list(y2)], dtype=float)
    try:
        return krippendorff.alpha(data, level_of_measurement=level)
    except Exception:
        return None


def icc_two_way(pairs):
    """ICC(2,1), two-way random effects, single measure."""
    if len(pairs) < 2: return None
    data = np.array(pairs, dtype=float)
    n, k = data.shape
    grand_mean    = data.mean()
    subject_means = data.mean(axis=1)
    rater_means   = data.mean(axis=0)
    SS_subjects = k * ((subject_means - grand_mean) ** 2).sum()
    SS_raters   = n * ((rater_means   - grand_mean) ** 2).sum()
    SS_total    = ((data - grand_mean) ** 2).sum()
    SS_error    = SS_total - SS_subjects - SS_raters
    MS_subjects = SS_subjects / (n - 1)
    MS_raters   = SS_raters   / (k - 1)
    MS_error    = SS_error    / ((n - 1) * (k - 1))
    denom = MS_subjects + (k - 1) * MS_error + k * (MS_raters - MS_error) / n
    if denom == 0: return None
    return (MS_subjects - MS_error) / denom


def f1_score(pairs):
    """F1 with pair[0] as gold and pair[1] as predicted, positive class = 1.

    Binary dimensions only.
    """
    if not pairs: return None
    tp = sum(g == 1 and p == 1 for g, p in pairs)
    fp = sum(g == 0 and p == 1 for g, p in pairs)
    fn = sum(g == 1 and p == 0 for g, p in pairs)
    if (tp + fp) == 0 or (tp + fn) == 0: return None
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    if precision + recall == 0: return None
    return 2 * precision * recall / (precision + recall)


def compute_row(pairs, binary=False, min_pairs=MIN_PAIRS):
    """Every metric appropriate to the scale, as one dict. None where N is too small."""
    n = len(pairs)
    if n < min_pairs:
        return {'n': n, 'em': None, 'w1': None, 'mae': None,
                'kappa': None, 'alpha': None, 'icc': None, 'f1': None}
    return {
        'n':     n,
        'em':    exact_match(pairs),
        'w1':    None if binary else within_one(pairs),
        'mae':   None if binary else mae(pairs),
        'kappa': cohen_kappa(pairs) if binary else None,
        'alpha': alpha(pairs, binary=binary),
        'icc':   None if binary else icc_two_way(pairs),
        'f1':    f1_score(pairs) if binary else None,
    }


# -- Loading -------------------------------------------------------------------

def load_annotations(tasks=TASKS):
    """Return (all_data, annotators).

    all_data:   task -> annotator -> {safe_instance_id: {dimension: value}}
    annotators: task -> [annotator, ...], 'gold' excluded
    """
    all_data, annotators = {}, {}
    for task in tasks:
        df   = load(task)
        dims = DIMENSIONS[task]
        anns = [a for a in annotators_for(df, dims[0]) if a != 'gold']
        annotators[task] = anns
        all_data[task] = {}
        for ann in anns:
            cols = {d: f'{d}_{ann}' for d in dims if f'{d}_{ann}' in df.columns}
            sub  = df[['safe_instance_id'] + list(cols.values())]
            sub  = sub.dropna(subset=list(cols.values()), how='all')
            records = {}
            for row in sub.to_dict('records'):
                entry = {}
                for dim, col in cols.items():
                    val = row[col]
                    if pd.isna(val):
                        continue
                    entry[dim] = int(val) if TASK_FORMAT[task] == 'likert' else val
                records[row['safe_instance_id']] = entry
            all_data[task][ann] = records
    return all_data, annotators


def corpus_order():
    """safe_instance_ids in queue order -- gives every table a stable row order."""
    return list(load('corpus')['safe_instance_id'])


# -- Pair collection -----------------------------------------------------------

def collect_er_pairs(er1, er2, order=None):
    """Event-relation value pairs between two annotators.

    Returns (span1_is_event, span2_is_event, temporal, temporal_binary,
    causality, causality_binary) pair lists.
    """
    shared = set(er1) & set(er2)
    order = order if order is not None else list(shared)
    sp1_ev, sp2_ev = [], []
    temp_pairs, temp_binary_pairs = [], []
    caus_pairs, caus_binary_pairs = [], []
    for inst_id in order:
        if inst_id not in shared: continue
        a1, a2 = er1.get(inst_id), er2.get(inst_id)
        if a1 is None or a2 is None: continue
        sp1_ev.append((int(bool(a1.get('span1_is_event'))), int(bool(a2.get('span1_is_event')))))
        sp2_ev.append((int(bool(a1.get('span2_is_event'))), int(bool(a2.get('span2_is_event')))))
        if a1.get('span1_is_event') and a1.get('span2_is_event') and \
           a2.get('span1_is_event') and a2.get('span2_is_event'):
            t1, t2 = a1.get('temporal_order'), a2.get('temporal_order')
            c1, c2 = a1.get('causality_rating'), a2.get('causality_rating')
            if t1 is not None and t2 is not None:
                temp_pairs.append((t1, t2))
                temp_binary_pairs.append((
                    int(t1 in ('span1_first', 'span2_first')),
                    int(t2 in ('span1_first', 'span2_first')),
                ))
            # Causal direction is undefined when either annotator called the
            # events simultaneous, so those instances are excluded.
            if c1 is not None and c2 is not None and \
               t1 != 'simultaneous' and t2 != 'simultaneous':
                caus_pairs.append((c1, c2))
                caus_binary_pairs.append((
                    int(c1 != 'not_related'),
                    int(c2 != 'not_related'),
                ))
    return sp1_ev, sp2_ev, temp_pairs, temp_binary_pairs, caus_pairs, caus_binary_pairs


def er_metric_pairs(er1, er2, order=None):
    """The five event-relation comparisons, keyed by the label they report under.

    Named rather than positional: the tuple from collect_er_pairs is easy to
    index off by one, and every one of these is a nominal/binary comparison, so
    a mis-indexed row still computes and just reports the wrong number.
    """
    sp1, sp2, temporal, temporal_binary, causality, causality_binary = \
        collect_er_pairs(er1, er2, order)
    return {
        'Span is event':                      sp1 + sp2,
        'Temporal order (nominal)':           temporal,
        'Temporal order (directional / not)': temporal_binary,
        'Causality (nominal)':                causality,
        'Causality (causal / not)':           causality_binary,
    }


def collect_likert_pairs(data, ann1, ann2, dim, order=None):
    """(value, value) pairs on one dimension for instances both annotators rated."""
    d1, d2 = data.get(ann1, {}), data.get(ann2, {})
    shared = set(d1) & set(d2)
    order = order if order is not None else list(shared)
    pairs = []
    for inst_id in order:
        if inst_id not in shared: continue
        v1 = (d1.get(inst_id) or {}).get(dim)
        v2 = (d2.get(inst_id) or {}).get(dim)
        if v1 is not None and v2 is not None:
            pairs.append((v1, v2))
    return pairs


def short_dim(task, dim):
    """'setting_temporal_grounding' -> 'Temporal Grounding'."""
    return dim.replace(f'{task}_', '').replace('_', ' ').title()


def agreement_rows(task, all_data, annotators, order=None, pooled=None, min_pairs=MIN_PAIRS):
    """Metric rows for every annotator pair on a task.

    One dict per (annotator pair, dimension), carrying task / dim / ann_pair
    alongside the metrics. `pooled` optionally adds one extra comparison of a
    reference annotator against several others combined, as
    ('tejo9855', ['roda9210', 'maria']) -- only valid when those others worked
    disjoint instance sets, so nothing is double-counted.

    This is the single source of truth for both the global summary and the
    per-task tables, which previously each collected pairs their own way.
    """
    from itertools import combinations

    order = order if order is not None else corpus_order()
    data  = all_data[task]
    anns  = [a for a in annotators[task] if data.get(a)]
    rows  = []

    if TASK_FORMAT[task] == 'event_relation':
        for ann1, ann2 in combinations(anns, 2):
            for label, pairs in er_metric_pairs(data[ann1], data[ann2], order).items():
                row = compute_row(pairs, binary=True, min_pairs=min_pairs)
                row.update(task=TASK_LABEL[task], dim=label, ann_pair=f'{ann1} / {ann2}')
                rows.append(row)
        return rows

    for ann1, ann2 in combinations(anns, 2):
        for dim in DIMENSIONS[task]:
            row = compute_row(collect_likert_pairs(data, ann1, ann2, dim, order),
                              min_pairs=min_pairs)
            row.update(task=TASK_LABEL[task], dim=short_dim(task, dim),
                       ann_pair=f'{ann1} / {ann2}')
            rows.append(row)

    if pooled:
        ref, others = pooled
        others = [o for o in others if data.get(o)]
        if data.get(ref) and others:
            ann_pair = f'{ref} / {"+".join(others)}'
            for dim in DIMENSIONS[task]:
                pairs = []
                for other in others:
                    pairs += collect_likert_pairs(data, ref, other, dim, order)
                row = compute_row(pairs, min_pairs=min_pairs)
                row.update(task=TASK_LABEL[task], dim=short_dim(task, dim),
                           ann_pair=ann_pair)
                rows.append(row)
    return rows
