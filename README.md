# TemporalAnalysis — CRAFT

Code, annotated dataset and model outputs for the CRAFT project: temporally ordering
adverse events reported in VAERS (Vaccine Adverse Event Reporting System) free-text
narratives with large language models.

Given a VAERS report and the list of adverse events coded for it, a model must arrange
the events into an ordered sequence of **stages**. Events with no stated temporal
relation share a stage; events that the narrative says appeared later go into a later
stage. Each stage is a JSON object mapping an adverse-event term to the phrase in the
narrative that supports it (or `"none"` when the term is not mentioned).

This branch (`chengyang`) holds the modelling, evaluation and analysis code.
The data preprocessing and annotation pipeline (SQL import scripts, sampling, o4-mini
annotation, human verification) lives on the `tahreem-vaers-analysis` branch.

## Repository layout

```
src/
  medtempo_eval.py        Evaluation library (metrics + MedTempoEvaluator API)
  utils.py                Small parsing helpers shared by the notebooks
notebooks/
  annotation/             o4-mini annotation of the sampled reports
  baselines/              Zero-shot / few-shot baselines (OpenAI, Claude, Llama, MedGemma)
  temporal_gating/        "Does this report contain temporal progression?" gate
  craft/                  CRAFT generate-verify-revise agent, all backbones and variants
  evaluation/             temp_eval.ipynb: metrics, tables and stage-count plots
  analysis/               final_eda.ipynb: EDA, agreement rates, symptom heatmaps
data/                     Dataset (see below)
results/                  Final model outputs used in the paper, gzipped (see below)
figures/                  Figures produced by the evaluation and analysis notebooks
requirements.txt
```

## Dataset (`data/`)

| File | Rows | Description |
|---|---|---|
| `VAERS_Sampled_Data_v4.csv` | 6000 | Sampled VAERS reports (1500 per manufacturer) with metadata: `vaers_id`, `vax_type`, `vax_manu`, `age_yrs`, `sex`, `vax_year`, `symptom_text`, `symptom_list`, `num_symptoms`, `report_length`, `length_category`. |
| `o4mini_annotation_final.csv` | 6000 | Gold annotation. `llm_result` / `llm_result_formatted` is the o4-mini draft; `t_*` and `h_*` are the two human annotators' corrections and notes; `final_result` / `final_corrected_version` is the adjudicated gold timeline; `agreement` and `llm_a*_agreement` are agreement flags. |
| `final_data.csv` | 6000 | `o4mini_annotation_final.csv` joined with the report metadata; input to `notebooks/analysis/final_eda.ipynb`. |
| `openai_gold_with_temporal.csv` | 3156 | Evaluation set: the subset of reports whose gold timeline contains temporal progression (more than one stage). All result files in `results/` are keyed to these `vaers_id`s. |
| `gold__num_stages.csv` | 3156 | Number of gold stages per evaluation report. |
| `samples/sample_40.csv`, `samples/sample_our_method_*.csv` | 40 / 300 | Development samples used while designing prompts (300 = 200 with temporal information + 100 without). |

The timeline columns are JSON strings: a list of stages, each a dict
`{"<adverse event>": ["<supporting phrase>", ...] | "none"}`.

`vaers_id` values are the public VAERS identifiers. Raw VAERS files are not included;
the `tahreem-vaers-analysis` branch has the import and sampling scripts.

## CRAFT method and variants

Every CRAFT run is a **generate → format-check → verify → revise** loop
(`TemporalExtractionAgent` in the `notebooks/craft/*` notebooks, `max_iterations = 4`):

1. **Generate**: the backbone LLM orders the given adverse events into stages.
2. **Format check**: deterministic repair (drop invented terms, add missing ones as
   `"none"`, canonicalise keys).
3. **Verify**: an LLM verifier scores the timeline 0–5 and returns feedback. Two
   verifier prompts are used: an *anchor* prompt that infers an explicit time anchor
   per stage and only checks chronology, over-merge and over-split against those
   anchors, and a general (*compass*) prompt.
4. **Revise**: if the score is below threshold, the feedback is fed back for another
   attempt. In the *inform* variants the feedback is free text given to the same
   generator; in the *reform* variants the verifier emits structured edit operations
   (`ASPECT=...; OP=...`) that a second generator applies to the previous timeline.

Method names in the paper vs. the names used in the notebooks and file names:

| Paper | Notebook / file name fragment |
|---|---|
| zero-shot baseline | `baseline` |
| inform + compass verifier | `plain` |
| inform + anchor verifier | `vanilla_anchor_updated` |
| reform + compass verifier | `cascade_vanilla` |
| reform + anchor verifier | `premium_updated` |
| temporal gate | `temporal_gate` |

Backbones: GPT-4.1 (`openai_*`), Claude Sonnet 4.5 (`claude_*`),
Llama-3.3-70B-Instruct (`llama70b_*`), MedGemma-27B (`medgemma_*`).

## Results (`results/`)

`results/<backbone>/<file>.csv.gz` holds the final output of every backbone × method
combination reported in the paper, plus the temporal-gate outputs. `results/file_map.json`
maps `<backbone>/<paper method>` to the file name. Each file has the evaluation-set
columns plus `new_llm_result` (or `hf_result`: the full attempt history as JSON) and
`final_result_cleaned` (the timeline that is scored).

Files are gzipped to keep the repository small; `pandas.read_csv` reads them directly.
Intermediate checkpoints (`*_part_*.csv`) and per-iteration dumps (`*_iter.csv`) are
not tracked.

## Evaluation (`src/medtempo_eval.py`)

Metrics implemented for bucketed (stage) sequences, all tie-aware:

- **Strict exact match** of the stage sequence (within-stage order ignored).
- **Group-aware LCCS**: longest run of consecutive stages shared by gold and prediction,
  with recall / precision / F1 over the number of stages.
- **Position error distance**: mean normalised |stage(pred) − stage(gold)| per event.
- **Item-set P/R/F1** over the events placed.
- **Kendall's τ-b** with ties on the intersection of events.

```python
import sys; sys.path.append("src")
from medtempo_eval import MedTempoEvaluator
ev = MedTempoEvaluator()
report = ev.evaluate_batch(gold_timelines, pred_timelines)   # lists of JSON strings or lists
print(report["macro"])
```

Inputs may be JSON strings, Python-literal strings, lists of dicts, or `"A,B | C"` text;
see the module docstring.

## Running the notebooks

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # notebooks/craft/openai_*, baselines, gating
export ANTHROPIC_API_KEY=...     # notebooks/craft/claude_*, baselines, gating
```

The notebooks were run with every data and result file in the working directory, so
their `read_csv` / `to_csv` calls use bare file names. To re-run one, either launch
Jupyter from a directory that contains the files it needs (copy or symlink them from
`data/` and `results/`, gunzipping the results) or prefix the paths. The open-weight
notebooks (`hf_*`, `open_source_hf_*`) need a GPU; MedGemma-27B and Llama-3.3-70B were
run with 8-bit loading on 2× 24 GB GPUs.

Outputs of `open_source_hf_med_gpu_0/1.ipynb` were cleared before committing (they
were 22 MB each); the corresponding results are in `results/medgemma/`.

## License

MIT, see `LICENSE`.
