#!/usr/bin/env python3
"""Build the MedTempo evaluation splits from the annotation file.

Reads
  data/o4mini_annotation_final.csv   6000 annotated reports
  data/VAERS_Sampled_Data_v4.csv     the same reports with vaccine metadata

Writes
  data/openai_gold_with_temporal.csv MedTempo-T: reports whose gold timeline has
                                     temporal progression (3166; the file name is
                                     kept because every notebook reads it)
  data/medtempo_NT.csv               MedTempo-NT: reports whose gold timeline is
                                     "No temporal information found" (2181)
  data/gold__num_stages.csv          number of stage groups per MedTempo-T report

Rules (match Section 3 of the paper: 6000 sampled -> 653 excluded for temporal
ambiguity -> 5347 MedTempo = 3166 T + 2181 NT):
  * final_result == 2 marks a report excluded by the adjudicators.
  * gold timeline = final_corrected_version when the annotators corrected the
    o4-mini draft, otherwise llm_result_formatted (the accepted draft).
  * a report is in T unless its gold timeline is "No temporal information found".
"""
from __future__ import annotations

import json
import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GOLD_COLS = ["vaers_id", "symptom_text", "symptom_list", "llm_result_formatted",
             "final_corrected_version", "final_result"]
NO_TEMPORAL = "no temporal information"


def num_groups(timeline: str) -> int:
    """Number of stage groups in a gold timeline JSON string (all groups count,
    including a trailing group of unmentioned terms, matching the original
    gold__num_stages.csv)."""
    for loader in (json.loads, ast.literal_eval,
                   lambda s: json.loads(re.sub(r",\s*([}\]])", r"\1", s))):
        try:
            parsed = loader(timeline)
            return len(parsed) if isinstance(parsed, list) else 0
        except Exception:
            continue
    return 0


def main() -> None:
    annot = pd.read_csv(DATA / "o4mini_annotation_final.csv", low_memory=False)
    vaers = pd.read_csv(DATA / "VAERS_Sampled_Data_v4.csv", low_memory=False)
    vaers["vaers_id"] = vaers["vaers_id"].astype(float).astype(int)
    vaers["vax_manu"] = vaers["vax_manu"].astype(str).str.strip()
    annot["vaers_id"] = annot["vaers_id"].astype(int)

    gold_timeline = annot["final_corrected_version"].where(
        annot["final_corrected_version"].notna(), annot["llm_result_formatted"])
    no_temporal = gold_timeline.astype(str).str.lower().str.contains(NO_TEMPORAL)
    kept = annot["final_result"] != 2

    T = annot[kept & ~no_temporal]
    NT = annot[kept & no_temporal]
    manu = vaers.set_index("vaers_id")["vax_manu"]

    T[GOLD_COLS].to_csv(DATA / "openai_gold_with_temporal.csv", index=False)
    NT[GOLD_COLS].to_csv(DATA / "medtempo_NT.csv", index=False)
    stages = pd.DataFrame({"vaers_id": T["vaers_id"],
                           "num_stages": gold_timeline[T.index].apply(num_groups)})
    stages.to_csv(DATA / "gold__num_stages.csv", index=False)

    print(f"sampled reports      : {len(annot)}")
    print(f"excluded (ambiguous) : {int((~kept).sum())}")
    print(f"MedTempo (T + NT)    : {len(T) + len(NT)}")
    print(f"MedTempo-T           : {len(T)}  {T['vaers_id'].map(manu).value_counts().to_dict()}")
    print(f"MedTempo-NT          : {len(NT)}  {NT['vaers_id'].map(manu).value_counts().to_dict()}")
    print(f"T stage groups       : mean {stages['num_stages'].mean():.2f}, "
          f"min {stages['num_stages'].min()}, max {stages['num_stages'].max()}")


if __name__ == "__main__":
    main()
