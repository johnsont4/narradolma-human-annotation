# narradolma-human-annotation

Analysis of the human annotation layer for the narratives-in-pretraining-data
project: inter-annotator agreement, label distributions, correlations with
automatic text features, and the public HuggingFace release.

Annotation itself happens in [potato](https://github.com/johnsont4/potato)
(`teagan_tasks/`). This repo never reads potato's raw output — it reads parquet.

## How data gets here

```
potato/teagan_tasks/
  <task>/annotation_output/results/<annotator>/user_state.json   raw potato output
  <task>/data/…_safeid_with_spans.csv                            source corpus
        │
        │   python export_annotations.py
        ▼
  narradolma-human-annotation/data/*.parquet      ← this repo
  llm-narrative-annotations/annotated_data/       ← LLM-vs-human comparison
```

To refresh after a round of annotation:

```bash
cd ../potato/teagan_tasks
python export_annotations.py           # writes to both destinations
python export_annotations.py --dry-run # preview without writing
```

New annotators are picked up automatically — the exporter discovers them from
the results directories rather than a hardcoded list.

## Data

Five parquet files in `data/`, all keyed on `safe_instance_id`:

| File | Rows | What |
|---|---|---|
| `corpus.parquet` | 1072 | Every sampled passage + metadata. No labels. |
| `setting_annotations.parquet` | 400 | Setting task, gold-annotated instances |
| `agency_annotations.parquet` | 405 | Agency task |
| `event_relation_annotations.parquet` | 440 | Event-relation task |
| `all_annotations.parquet` | 457 | The three tasks outer-joined |

Label columns follow `narradolma-catalog/SCHEMA.md`:

| Pattern | Meaning | Example |
|---|---|---|
| `{dim}_gold` | adjudicated label | `agency_conflict_gold` |
| `{dim}_{annotator}` | one annotator's label | `setting_sensory_roda9210` |
| `annotation_order_{annotator}` | position in that annotator's queue | used by the drift analysis |

Nine Likert dimensions (1–5), plus four event-relation dimensions:

```
agency:          focalization, emotion, cognition, change_of_state, conflict
setting:         concreteness, temporal_grounding, spatial_grounding, sensory
event_relation:  span1_is_event, span2_is_event, temporal_order, causality_rating
```

`tejo9855` is the adjudicator, so `_gold` duplicates `_tejo9855`. Every other
annotator worked on a strict subset of the gold instances.

## Layout

```
data/         parquet, written by potato's export_annotations.py
notebooks/    analysis; nb_utils.load() is the only data entry point
features/     extract_features.py -> features.csv (spaCy, VADER, concreteness)
figures/      save_*.py -> figures/pdfs/, plus the paper's .tex table
release/      export_release.py (anonymize + trim) -> upload_to_hf.py
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # features/ only
python -m nltk.downloader vader_lexicon      # features/ only
```

## Common tasks

```bash
# Regenerate the paper's distribution figures
python figures/save_distribution_pdfs.py
python figures/save_combined_distribution.py

# Recompute automatic text features (slow; needs spaCy + NLTK)
python features/extract_features.py

# Build and inspect the public release, then push it
python release/export_release.py
python release/upload_to_hf.py               # dry run, prints what it would do
python release/upload_to_hf.py --yes         # actually pushes to HuggingFace
```

`release/export_release.py` anonymizes annotators (`maria` → `annotator_1`,
`roda9210` → `annotator_2`) and drops everyone not in its `ANON_MAP`. Anyone
added to the annotation effort must be added there explicitly before they appear
in a public release.
