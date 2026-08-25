"""
agreement_report.py

HTML rendering for the agreement notebook: metric tables and disagreement
cards. Kept apart from agreement.py so the metrics stay free of presentation,
and out of the notebook so the notebook is analysis rather than markup.

    from agreement_report import AgreementReport

    rep = AgreementReport()
    rep.summary()                      # per-task averages
    rep.table('setting')               # every pair, every dimension
    rep.disagreements('agency')        # worst-first example cards

Everything is loaded once when the report is constructed.
"""

import html
import json
from collections import defaultdict

from IPython.display import display, HTML

from nb_utils import load, DIMENSIONS, TASKS
from agreement import (
    TASK_LABEL, TASK_FORMAT, MIN_PAIRS,
    load_annotations, corpus_order, agreement_rows, er_metric_pairs,
)

TASK_BG = {
    'Event Relation': '#e8f4fd',
    'Agency':         '#fdf3e8',
    'Setting':        '#edf8ee',
}

# Setting's two non-overlapping partners can be pooled against the adjudicator
# without double-counting any instance.
POOLED = {'setting': ('tejo9855', ['roda9210', 'maria'])}

DASH = '<span style="color:#bbb">—</span>'

_TH = 'padding:6px 12px'
_TD = 'padding:6px 12px;text-align:center'

METRIC_COLUMNS = [
    ('Exact match',       'em',    True),
    ('Within 1',          'w1',    True),
    ('MAE',               'mae',   False),
    ("Krippendorff's α",  'alpha', False),
    ("Cohen's κ",         'kappa', False),
    ('ICC(2,1)',          'icc',   False),
    ('F1 (gold=tejo9855)','f1',    False),
]

NOTES = (
    '<p style="color:#777;font-size:0.8em;margin-top:10px;line-height:1.6">'
    '<b>Within 1</b>: fraction of pairs differing by at most 1 point (ordinal only).<br>'
    '<b>MAE</b>: mean absolute error (ordinal only).<br>'
    "<b>Krippendorff's α</b>: nominal metric for binary/categorical labels; "
    'interval metric for ordinal scales.<br>'
    "<b>Cohen's κ</b>: chance-corrected agreement for nominal/binary labels.<br>"
    '<b>ICC(2,1)</b>: two-way random effects; credits agreement even when one '
    'annotator rates systematically higher or lower (ordinal only).<br>'
    '<b>F1</b>: binary dimensions only; tejo9855 = gold, positive class = 1.<br>'
    '<b>pooled rows</b> (<i>a / b+c</i>): b and c annotated disjoint instance '
    'sets, so pooling them double-counts nothing.<br>'
    f'Rows with N &lt; {MIN_PAIRS} show — (too few shared instances).'
    '</p>'
)


def fmt(v, pct=False):
    """A metric as text; an em dash when it could not be computed."""
    if v is None: return DASH
    return f'{v*100:.1f}%' if pct else f'{v:.3f}'


def _metric_cells(row):
    return ''.join(f'<td style="{_TD}">{fmt(row[key], pct=pct)}</td>'
                   for _, key, pct in METRIC_COLUMNS)


class AgreementReport:
    """Loads the annotations once, then renders any view of them."""

    def __init__(self, tasks=TASKS):
        self.tasks = list(tasks)
        self.all_data, self.annotators = load_annotations(self.tasks)
        self.order = corpus_order()
        self.corpus_rows = {r['safe_instance_id']: r
                            for r in load('corpus').to_dict('records')}

    # -- metric rows ----------------------------------------------------------

    def rows(self, task):
        return agreement_rows(task, self.all_data, self.annotators,
                              self.order, pooled=POOLED.get(task))

    def all_rows(self):
        return [r for task in self.tasks for r in self.rows(task)]

    # -- tables ---------------------------------------------------------------

    def table(self, task=None, title=None):
        """Every annotator pair × dimension. One task, or all of them stacked."""
        rows = self.rows(task) if task else self.all_rows()
        show_task_col = task is None

        header = '<tr style="background:#f0f0f0">'
        if show_task_col:
            header += f'<th style="{_TH};text-align:left">Task</th>'
        header += f'<th style="{_TH};text-align:left">Dimension</th>'
        header += f'<th style="{_TH}">N</th>'
        header += ''.join(f'<th style="{_TH}">{label}</th>' for label, _, _ in METRIC_COLUMNS)
        header += '</tr>'

        ncols = len(METRIC_COLUMNS) + 2 + (1 if show_task_col else 0)
        body, prev_task, prev_pair = '', None, None
        for row in rows:
            if row['ann_pair'] != prev_pair:
                body += (f'<tr style="background:#f8f8f8;border-top:2px solid #ccc">'
                         f'<td colspan="{ncols}" style="padding:3px 12px;font-size:0.8em;'
                         f'color:#555;font-style:italic">↳ {row["ann_pair"]}</td></tr>')
            prev_pair = row['ann_pair']
            bg = TASK_BG.get(row['task'], '#fff')
            body += '<tr style="border-top:1px solid #eee">'
            if show_task_col:
                weight = ';font-weight:bold' if row['task'] != prev_task else ''
                text   = row['task'] if row['task'] != prev_task else ''
                body += f'<td style="{_TH};background:{bg}{weight}">{text}</td>'
                prev_task = row['task']
            body += f'<td style="{_TH}">{row["dim"]}</td>'
            body += f'<td style="{_TD}">{row["n"]}</td>'
            body += _metric_cells(row)
            body += '</tr>'

        heading = title or (f'{TASK_LABEL[task]} agreement' if task else 'All tasks')
        display(HTML(
            '<div style="font-family:sans-serif">'
            f'<b style="font-size:1.05em">{heading}</b>'
            '<table style="border-collapse:collapse;font-size:0.92em;margin-top:8px">'
            f'<thead>{header}</thead><tbody>{body}</tbody></table>'
            + NOTES + '</div>'
        ))

    def summary(self):
        """One row per task × annotator pair: the averaged headline metrics."""
        def mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        groups = defaultdict(list)
        for row in self.all_rows():
            groups[(row['task'], row['ann_pair'])].append(row)

        header = ('<tr style="background:#f0f0f0">'
                  f'<th style="{_TH};text-align:left">Task</th>'
                  f'<th style="{_TH};text-align:left">Annotators</th>'
                  f'<th style="{_TH}">Avg α</th><th style="{_TH}">Avg MAE</th>'
                  f'<th style="{_TH}">Avg κ</th><th style="{_TH}">Avg F1</th></tr>')

        body, prev_task = '', None
        for (task, ann_pair), rows in groups.items():
            ordinal = task in ('Setting', 'Agency')
            cells = [
                mean([r['alpha'] for r in rows]) if ordinal else None,
                mean([r['mae']   for r in rows]) if ordinal else None,
                None if ordinal else mean([r['kappa'] for r in rows]),
                None if ordinal else mean([r['f1']    for r in rows]),
            ]
            bg = TASK_BG.get(task, '#fff')
            weight = ';font-weight:bold' if task != prev_task else ''
            label  = task if task != prev_task else ''
            prev_task = task
            body += ('<tr style="border-top:1px solid #eee">'
                     f'<td style="{_TH};background:{bg}{weight}">{label}</td>'
                     f'<td style="{_TH};font-size:0.85em;color:#555">{ann_pair}</td>'
                     + ''.join(f'<td style="{_TD}">{fmt(v)}</td>' for v in cells)
                     + '</tr>')

        display(HTML(
            '<div style="font-family:sans-serif;margin-top:16px">'
            '<b style="font-size:1.05em">Per-task average metrics</b>'
            '<table style="border-collapse:collapse;font-size:0.92em;margin-top:8px">'
            f'<thead>{header}</thead><tbody>{body}</tbody></table>'
            '<p style="color:#777;font-size:0.8em;margin-top:6px">'
            'Averaged across dimensions. Setting &amp; Agency report α (interval) '
            'and MAE; Event Relation reports κ (nominal) and F1.</p></div>'
        ))

    # -- disagreement cards ---------------------------------------------------

    def disagreements(self, task, min_diff=1, max_show=50,
                      ann1=None, ann2=None, feature=None):
        """Instances the annotators disagreed on, worst first.

        ann1/ann2 restrict to one pair; feature restricts to one dimension
        (short or full name, e.g. 'concreteness' or 'setting_concreteness').
        """
        from itertools import combinations

        anns = [a for a in self.annotators[task] if self.all_data[task].get(a)]
        pairs = list(combinations(anns, 2))
        if ann1 and ann2:
            target = tuple(sorted([ann1, ann2]))
            pairs = [p for p in pairs if tuple(sorted(p)) == target]
            if not pairs:
                return display(HTML(
                    f'<p style="color:#c00">Pair ({ann1}, {ann2}) not found in '
                    f'{TASK_LABEL[task]}. Available: {anns}</p>'))
        if not pairs:
            return display(HTML('<p style="color:#888">Need ≥ 2 annotators with '
                                f'data for {TASK_LABEL[task]}.</p>'))

        dims = DIMENSIONS[task]
        likert = TASK_FORMAT[task] == 'likert'

        feature_dim = None
        if feature is not None:
            matches = [d for d in dims
                       if feature.lower() in (d.lower(), d.replace(f'{task}_', '').lower())]
            if not matches:
                return display(HTML(
                    f'<p style="color:#c00">Feature "{feature}" not found in '
                    f'{TASK_LABEL[task]}. Available: '
                    f'{[d.replace(f"{task}_", "") for d in dims]}</p>'))
            feature_dim = matches[0]

        out = '<div style="font-family:sans-serif">'
        for a1, a2 in pairs:
            records = (self._likert_records(task, a1, a2, anns, min_diff, feature_dim)
                       if likert else
                       self._er_records(task, a1, a2, anns, feature_dim))
            records.sort(key=lambda r: r[0], reverse=True)
            records = records[:max_show]

            out += (f'<b style="font-size:1.05em">{TASK_LABEL[task]} — disagreements'
                    + (f' (|diff| ≥ {min_diff})' if likert else '')
                    + f', top {len(records)}</b>'
                    + (f' · feature = <b>{feature}</b>' if feature else '')
                    + f' <span style="color:#888;font-size:0.9em">{a1} vs {a2}</span>')
            if not records:
                out += '<p style="color:#888;margin-left:10px">No disagreements found.</p>'
            else:
                out += ('<br><small style="color:#999">green = agree · yellow = diff 1 · '
                        'orange = diff 2 · red = diff ≥ 3</small>' if likert else
                        '<br><small style="color:#999">'
                        f'<span style="{SPAN1_STYLE}">span 1</span> &nbsp; '
                        f'<span style="{SPAN2_STYLE}">span 2</span></small>')
                for sort_d, inst_id, data_rows in records:
                    out += self._card(inst_id, sort_d, a1, a2, anns, data_rows)
            out += '<br>'
        display(HTML(out + '</div>'))

    def _likert_records(self, task, a1, a2, anns, min_diff, feature_dim):
        dims   = DIMENSIONS[task]
        labels = [d.replace(f'{task}_', '').replace('_', ' ').title() for d in dims]
        d1, d2 = self.all_data[task][a1], self.all_data[task][a2]
        shared = set(d1) & set(d2)
        records = []
        for inst_id in self.order:
            if inst_id not in shared: continue
            v1, v2 = d1.get(inst_id) or {}, d2.get(inst_id) or {}
            diffs = [abs(v1[d] - v2[d]) if v1.get(d) is not None and v2.get(d) is not None
                     else None for d in dims]
            if feature_dim is not None:
                sort_d = diffs[dims.index(feature_dim)]
            else:
                sort_d = max((d for d in diffs if d is not None), default=0)
            if sort_d is None or sort_d < min_diff:
                continue
            data_rows = [
                (labels[i],
                 {ann: (self.all_data[task][ann].get(inst_id) or {}).get(dims[i])
                  for ann in anns},
                 diffs[i])
                for i in range(len(dims))
            ]
            records.append((sort_d, inst_id, data_rows))
        return records

    def _er_records(self, task, a1, a2, anns, feature_dim):
        er1, er2 = self.all_data[task][a1], self.all_data[task][a2]
        shared = set(er1) & set(er2)
        records = []
        for inst_id in self.order:
            if inst_id not in shared: continue
            g, p = er1.get(inst_id), er2.get(inst_id)
            if g is None or p is None: continue
            sp1_d = int(bool(g.get('span1_is_event')) != bool(p.get('span1_is_event')))
            sp2_d = int(bool(g.get('span2_is_event')) != bool(p.get('span2_is_event')))
            t1, t2 = g.get('temporal_order'), p.get('temporal_order')
            c1, c2 = g.get('causality_rating'), p.get('causality_rating')
            temp_d = None if t1 is None or t2 is None else int(t1 != t2)
            caus_d = None if c1 is None or c2 is None else int(c1 != c2)

            if feature_dim is not None:
                sort_d = {'span1_is_event': sp1_d, 'span2_is_event': sp2_d,
                          'temporal_order': temp_d, 'causality_rating': caus_d}[feature_dim]
                if not sort_d: continue
            else:
                sort_d = max(sp1_d, sp2_d, temp_d or 0, caus_d or 0)
                if not sort_d: continue

            def vals(key, transform=None):
                out = {}
                for ann in anns:
                    d = self.all_data[task][ann].get(inst_id)
                    v = None if d is None else d.get(key)
                    out[ann] = transform(v) if transform and v is not None else v
                return out

            yn = lambda v: 'Y' if v else 'N'
            data_rows = [('Span 1 event', vals('span1_is_event', yn), sp1_d),
                         ('Span 2 event', vals('span2_is_event', yn), sp2_d)]
            if temp_d is not None:
                data_rows.append(('Temporal order', vals('temporal_order'), temp_d))
            if caus_d is not None:
                data_rows.append(('Causality', vals('causality_rating'), caus_d))
            records.append((sort_d, inst_id, data_rows))
        return records

    def _card(self, inst_id, sort_d, a1, a2, anns, data_rows):
        return (
            '<div style="margin:14px 0;border:1px solid #ddd;border-radius:5px;'
            'overflow:hidden">'
            f'<div style="background:#f7f7f7;padding:5px 12px;font-size:0.78em;'
            f'color:#888;font-family:monospace">{inst_id} &nbsp;·&nbsp; '
            f'max |diff| = {sort_d}</div>'
            + self._text_card(inst_id)
            + _dim_table(a1, a2, anns, data_rows)
            + '</div>'
        )

    def _text_card(self, inst_id):
        row = self.corpus_rows.get(inst_id, {})
        return ('<div style="padding:14px 16px;font-size:0.93em;color:#222;'
                'line-height:1.85;border-bottom:1px solid #eee">'
                + highlighted_snippet(row.get('sampled_text', ''), row)
                + '</div>')


# -- span highlighting ---------------------------------------------------------

SPAN1_STYLE = ('background:#4e9af1;font-weight:600;border-radius:3px;'
               'padding:1px 4px;color:#fff')
SPAN2_STYLE = ('background:#f4a432;font-weight:600;border-radius:3px;'
               'padding:1px 4px;color:#fff')
CONTEXT = 150   # characters of surrounding text to keep around the spans


def parse_span(raw):
    """'[192, 199, "promote", "verb"]' -> (192, 199); None when unparseable."""
    if not raw: return None
    try:
        parsed = json.loads(raw)
        return int(parsed[0]), int(parsed[1])
    except (ValueError, TypeError, IndexError):
        return None


def highlighted_snippet(text, row):
    """A window of text around the assigned spans, with the spans marked up."""
    spans = []
    for key, style in (('assigned_span1', SPAN1_STYLE), ('assigned_span2', SPAN2_STYLE)):
        span = parse_span(row.get(key, ''))
        if span: spans.append((span[0], span[1], style))

    if spans:
        win_start = max(0, min(s[0] for s in spans) - CONTEXT)
        win_end   = min(len(text), max(s[1] for s in spans) + CONTEXT)
    else:
        win_start, win_end = 0, min(len(text), 400)

    chunk = text[win_start:win_end]
    parts, pos = [], 0
    for start, end, style in sorted(((s[0] - win_start, s[1] - win_start, s[2])
                                     for s in spans), key=lambda x: x[0]):
        start, end = max(pos, max(0, start)), min(len(chunk), end)
        if start >= end: continue
        parts.append(html.escape(chunk[pos:start]))
        parts.append(f'<span style="{style}">{html.escape(chunk[start:end])}</span>')
        pos = end
    parts.append(html.escape(chunk[pos:]))

    prefix = '<span style="color:#aaa">… </span>' if win_start > 0 else ''
    suffix = '<span style="color:#aaa"> …</span>' if win_end < len(text) else ''
    return prefix + ''.join(parts) + suffix


DIFF_BG = ['#d4edda', '#fff3cd', '#fde8d8', '#f8d7da', '#f5c6cb']


def _dim_table(a1, a2, anns, data_rows):
    header = ('<tr style="background:#f0f0f0">'
              '<th style="padding:4px 10px;text-align:left">Dimension</th>'
              + ''.join(f'<th style="padding:4px 10px">{a}</th>' for a in anns)
              + f'<th style="padding:4px 10px">|diff| ({a1} vs {a2})</th></tr>')
    body = ''
    for label, vals, d in data_rows:
        bg = '#f5f5f5' if d is None else DIFF_BG[min(d, len(DIFF_BG) - 1)]
        body += (f'<tr style="background:{bg};border-top:1px solid #eee">'
                 f'<td style="padding:4px 10px">{label}</td>'
                 + ''.join('<td style="padding:4px 10px;text-align:center">'
                           f'{vals.get(a) if vals.get(a) is not None else "—"}</td>'
                           for a in anns)
                 + '<td style="padding:4px 10px;text-align:center">'
                 + (str(d) if d is not None else '—') + '</td></tr>')
    return ('<table style="border-collapse:collapse;font-size:0.88em">'
            f'{header}{body}</table>')
