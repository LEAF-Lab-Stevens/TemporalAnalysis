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
  results/missing_from_results.txt   MedTempo-T ids that have no prediction in the
                                     result files shipped under results/

Rules (match Section 3 of the paper: 6000 sampled -> 653 excluded for temporal
ambiguity -> 5347 MedTempo = 3166 T + 2181 NT):
  * final_result == 2 marks a report excluded by the adjudicators.
  * gold timeline = final_corrected_version when the annotators corrected the
    o4-mini draft, otherwise llm_result_formatted (the accepted draft).
  * a report is in T unless its gold timeline is "No temporal information found".
The earlier notebook filter also tested llm_result_formatted on its own, which
dropped the 10 reports where o4-mini reported no progression but the annotators
corrected it to a multi-stage timeline (3156 instead of 3166).
"""
from __future__ import annotations

import gzip
import json
import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
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

    # Which MedTempo-T reports have no prediction in the shipped result files?
    missing: dict[int, set[str]] = {}
    for f in sorted(RESULTS.glob("*/*.csv.gz")):
        if "temporal_gate" in f.name:
            continue
        with gzip.open(f, "rt") as fh:
            ids = set(pd.read_csv(fh, usecols=["vaers_id"])["vaers_id"].astype(int))
        for vid in set(T["vaers_id"]) - ids:
            missing.setdefault(vid, set()).add(f.relative_to(RESULTS).as_posix())
    n_files = sum(1 for f in RESULTS.glob("*/*.csv.gz") if "temporal_gate" not in f.name)
    with open(RESULTS / "missing_from_results.txt", "w") as fh:
        fh.write("# MedTempo-T reports with no prediction in the result files under results/\n"
                 "# (they were dropped by the earlier temporal filter, see src/build_medtempo_splits.py)\n"
                 "# vaers_id\tvaccine\tfiles_missing_from\n")
        for vid in sorted(missing):
            fh.write(f"{vid}\t{manu.get(vid, '?')}\t{len(missing[vid])}/{n_files}\n")
    print(f"T ids missing from result files: {len(missing)} (listed in results/missing_from_results.txt)")


if __name__ == "__main__":
    main()
