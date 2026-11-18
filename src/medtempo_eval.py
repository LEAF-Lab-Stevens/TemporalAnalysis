#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MedTEMPO evaluation utilities.

This module packages the evaluation logic into an importable API.

Primary API:
  - MedTempoEvaluator: evaluate_sample / evaluate_batch / evaluate_from_dataframes

Input flexibility
-----------------
Gold/pred entries can be any of:
  - list[list[str]]: buckets/stages
  - list[dict]: each dict = a stage; keys are symptom codes; values can be any
               (values that are "none"/empty are ignored)
  - dict: either an envelope ("groups"/"buckets"/"timeline"/...) or a symptom->value map
  - JSON / Python-literal string of any of the above
  - simple text like: "A,B | C | D,E" (pipes separate buckets)

Outputs
-------
- evaluate_sample returns per-example metrics and canonicalized inputs.
- evaluate_batch returns per-example results plus macro averages and micro coverage.

Notes
-----
- Within-bucket order is ignored.
- Ties are represented as buckets.
- Kendall's tau-b is computed on the intersection of items.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import ast
import json
import math
import re
from collections import Counter


# ---------------- Normalization ----------------

def _normalize_default(x: Any) -> str:
    return str(x).strip().lower()


# ---------------- Metrics ----------------

def strict_exact_match(
    gold: Iterable[Iterable[Any]],
    pred: Iterable[Iterable[Any]],
    normalize: Optional[Callable[[Any], Any]] = None,
) -> Tuple[bool, str]:
    """Strict exact match for bucketed sequences.

    - Bucket order MUST match exactly.
    - Within-bucket order is ignored.
    - Bucket contents are treated as multisets (duplicate counts must match).
    """
    gold_buckets = [list(b) for b in gold]
    pred_buckets = [list(b) for b in pred]

    if len(gold_buckets) != len(pred_buckets):
        return False, f"Different # of buckets: gold={len(gold_buckets)} vs pred={len(pred_buckets)}"

    norm = (lambda x: normalize(x)) if normalize is not None else (lambda x: x)

    for i, (g_b, p_b) in enumerate(zip(gold_buckets, pred_buckets)):
        g_ctr = Counter(norm(x) for x in g_b)
        p_ctr = Counter(norm(x) for x in p_b)
        if g_ctr != p_ctr:
            missing = g_ctr - p_ctr
            extra = p_ctr - g_ctr
            parts = []
            if missing:
                parts.append(f"missing={dict(sorted(missing.items()))}")
            if extra:
                parts.append(f"extra={dict(sorted(extra.items()))}")
            return False, f"Bucket {i} differs: {'; '.join(parts) if parts else 'bucket mismatch'}"

    return True, ""


def group_aware_lccs(
    gold: Iterable[Iterable[Any]],
    pred: Iterable[Iterable[Any]],
    normalize: Optional[Callable[[Any], Any]] = None,
    return_alignment: bool = True,
) -> Tuple[int, float, float, float, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Group-aware Longest Common Contiguous Subsequence (LCCS) over buckets.

    Each bucket is treated as a set (order-insensitive).
    Returns longest run of consecutive buckets that appears in both sequences.
    """

    def canon_bucket(bucket: Iterable[Any]) -> Hashable:
        if normalize is None:
            return frozenset(bucket)
        return frozenset(normalize(x) for x in bucket)

    gold_tokens: List[Hashable] = [canon_bucket(b) for b in gold]
    pred_tokens: List[Hashable] = [canon_bucket(b) for b in pred]
    m, n = len(gold_tokens), len(pred_tokens)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best_len = 0
    best_end_g = 0
    best_end_p = 0

    for i in range(1, m + 1):
        gi = gold_tokens[i - 1]
        for j in range(1, n + 1):
            if gi == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_end_g = i
                    best_end_p = j
            else:
                dp[i][j] = 0

    lccs_len = best_len
    recall_gold = lccs_len / max(1, m)
    precision_pred = lccs_len / max(1, n)
    f1 = (2 * recall_gold * precision_pred / (recall_gold + precision_pred)) if (recall_gold + precision_pred) else 0.0

    if not return_alignment or lccs_len == 0:
        return lccs_len, recall_gold, precision_pred, f1, None

    g_start = best_end_g - lccs_len
    g_end = best_end_g - 1
    p_start = best_end_p - lccs_len
    p_end = best_end_p - 1
    return lccs_len, recall_gold, precision_pred, f1, ((g_start, g_end), (p_start, p_end))


def position_error_distance(
    gold: Iterable[Iterable[Any]],
    pred: Iterable[Iterable[Any]],
    normalize: Optional[Callable[[Any], Any]] = None,
    missing_penalty: str = "max",  # "ignore" | "max"
    return_details: bool = False,
) -> Tuple[float, float, Dict[str, Any]]:
    """Tie-aware position error distance for bucketed sequences.

    - Assign each item rank = bucket index.
    - Distance = |rank_pred - rank_gold|.
    - Normalize by max_dist = max(#buckets)-1, mapped to [0,1].
    - If missing_penalty == "max", penalize missing/extra items with worst-case distance.

    Returns (score, mae_norm, details) where score = 1 - mae_norm.
    """

    def norm(x: Any) -> Any:
        return x if normalize is None else normalize(x)

    def item_to_rank(seq: Iterable[Iterable[Any]]) -> Dict[Any, int]:
        rank_map: Dict[Any, int] = {}
        for b_idx, bucket in enumerate(seq):
            for x in bucket:
                nx = norm(x)
                if nx not in rank_map:
                    rank_map[nx] = b_idx
        return rank_map

    # materialize for bucket count
    gold_buckets: List[List[Any]] = [list(b) for b in gold]
    pred_buckets: List[List[Any]] = [list(b) for b in pred]

    gold_ranks = item_to_rank(gold_buckets)
    pred_ranks = item_to_rank(pred_buckets)

    B = max(len(gold_buckets), len(pred_buckets))
    max_dist = max(1, B - 1)

    gold_items = set(gold_ranks.keys())
    pred_items = set(pred_ranks.keys())
    inter = gold_items & pred_items
    only_gold = gold_items - pred_items
    only_pred = pred_items - gold_items

    distances: Dict[Any, int] = {}

    for x in inter:
        distances[x] = abs(pred_ranks[x] - gold_ranks[x])

    if missing_penalty == "max":
        for x in only_gold:
            distances[x] = max_dist
        for x in only_pred:
            distances[x] = max_dist
    elif missing_penalty == "ignore":
        pass
    else:
        raise ValueError('missing_penalty must be "ignore" or "max"')

    if len(distances) == 0:
        has_any_gold = any(len(b) > 0 for b in gold_buckets)
        has_any_pred = any(len(b) > 0 for b in pred_buckets)
        if not has_any_gold and not has_any_pred:
            score, mae_norm = 1.0, 0.0
            note = "both gold and pred empty; no displacement possible"
        else:
            score, mae_norm = 0.0, 1.0
            note = "no shared items; intersection-based position treated as worst"

        details = {} if not return_details else {
            "num_gold_buckets": len(gold_buckets),
            "num_pred_buckets": len(pred_buckets),
            "counts": {"intersection": 0, "only_gold": len(only_gold), "only_pred": len(only_pred)},
            "max_dist": max_dist,
            "mae_raw": None,
            "per_item_distance": {},
            "note": note,
        }
        return score, mae_norm, details

    mae = sum(distances.values()) / len(distances)
    mae_norm = min(1.0, mae / max_dist)
    score = max(0.0, 1.0 - mae_norm)

    details = {}
    if return_details:
        details = {
            "num_gold_buckets": len(gold_buckets),
            "num_pred_buckets": len(pred_buckets),
            "counts": {
                "intersection": len(inter),
                "only_gold": len(only_gold),
                "only_pred": len(only_pred),
                "evaluated_items": len(distances),
            },
            "max_dist": max_dist,
            "mae_raw": mae,
            "per_item_distance": dict(sorted(distances.items(), key=lambda kv: (-kv[1], str(kv[0])))),
            "note": None,
        }

    return score, mae_norm, details


def _item_set(seq: Iterable[Iterable[Any]], normalize: Optional[Callable[[Any], Any]]) -> set:
    norm = (lambda x: x) if normalize is None else normalize
    return {norm(x) for bucket in seq for x in bucket}


def _item_ranks(seq: Iterable[Iterable[Any]], normalize: Optional[Callable[[Any], Any]]) -> Dict[Any, int]:
    norm = (lambda x: x) if normalize is None else normalize
    ranks: Dict[Any, int] = {}
    for b_idx, bucket in enumerate(seq):
        seen = set()
        for x in bucket:
            k = norm(x)
            if k in seen:
                continue
            seen.add(k)
            if k not in ranks:
                ranks[k] = b_idx
    return ranks


def item_set_prf1(
    gold: Iterable[Iterable[Any]],
    pred: Iterable[Iterable[Any]],
    normalize: Optional[Callable[[Any], Any]] = None,
) -> Tuple[float, float, float, Dict[str, int]]:
    """Coverage-only precision/recall/F1 on unique items (order ignored)."""
    gold_items = _item_set(gold, normalize)
    pred_items = _item_set(pred, normalize)
    inter = gold_items & pred_items

    precision = len(inter) / len(pred_items) if pred_items else 1.0
    recall = len(inter) / len(gold_items) if gold_items else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    details = {
        "gold_count": len(gold_items),
        "pred_count": len(pred_items),
        "intersection": len(inter),
        "only_gold": len(gold_items - pred_items),
        "only_pred": len(pred_items - gold_items),
    }
    return precision, recall, f1, details


def kendall_tau_b_ties(
    gold: Iterable[Iterable[Any]],
    pred: Iterable[Iterable[Any]],
    normalize: Optional[Callable[[Any], Any]] = None,
    return_details: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Tie-aware Kendall's tau-b on bucket ranks over the intersection of items."""
    gold_r = _item_ranks(gold, normalize)
    pred_r = _item_ranks(pred, normalize)

    items = sorted(gold_r.keys() & pred_r.keys(), key=lambda x: str(x))
    n = len(items)
    if n < 2:
        return 1.0, ({"num_items_intersection": n, "pairs": 0} if return_details else {})

    C = D = T_g = T_p = 0
    disagreements: List[Tuple[Any, Any, int, int]] = []

    for i in range(n):
        a = items[i]
        ga = gold_r[a]
        pa = pred_r[a]
        for j in range(i + 1, n):
            b = items[j]
            gb = gold_r[b]
            pb = pred_r[b]

            rg = -1 if ga < gb else (1 if ga > gb else 0)
            rp = -1 if pa < pb else (1 if pa > pb else 0)

            if rg == 0:
                T_g += 1
            if rp == 0:
                T_p += 1

            if rg != 0 and rp != 0:
                if rg == rp:
                    C += 1
                else:
                    D += 1
                    if len(disagreements) < 20:
                        disagreements.append((a, b, rg, rp))

    numer = C - D
    denom_left = C + D + T_g
    denom_right = C + D + T_p
    denom = math.sqrt(denom_left * denom_right) if denom_left > 0 and denom_right > 0 else 0.0
    tau_b = (numer / denom) if denom > 0 else 1.0

    if not return_details:
        return tau_b, {}

    return tau_b, {
        "num_items_intersection": n,
        "pairs": n * (n - 1) // 2,
        "C": C,
        "D": D,
        "T_g": T_g,
        "T_p": T_p,
        "numerator": numer,
        "denominator_parts": {"left": denom_left, "right": denom_right},
        "tau_b": tau_b,
        "sample_disagreements": disagreements,
        "relation_legend": {"-1": "a precedes b", "0": "tie", "+1": "a follows b"},
    }


# ---------------- Parsing/coercion ----------------

def _coerce_to_buckets(raw: Any, normalize: Optional[Callable[[Any], Any]] = None) -> List[List[Any]]:
    """Convert raw model outputs into List[List[item]] buckets."""
    norm = (lambda x: x) if normalize is None else normalize

    def _dedupe_bucket(b: Iterable[Any]) -> List[Any]:
        seen, out = set(), []
        for x in b:
            k = norm(x)
            if k in seen or k == "":
                continue
            seen.add(k)
            out.append(k)
        return out

    def _is_none_val(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and norm(v) == "none":
            return True
        if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
            return True
        return False

    # ---------- STRING INPUT ----------
    if isinstance(raw, str):
        s = raw.strip()

        # Try JSON or python literal
        if s.startswith("{") or s.startswith("[") or s.startswith("```"):
            try:
                obj = json.loads(s.strip("`"))
                return _coerce_to_buckets(obj, normalize=normalize)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                    return _coerce_to_buckets(obj, normalize=normalize)
                except Exception:
                    pass

        # Fallback: 'A,B | C | D,E'
        groups = re.split(r"\|\||\||\n+", s)
        buckets: List[List[Any]] = []
        for g in groups:
            toks = [t.strip() for t in re.split(r"[,\t;]+", g) if t.strip()]
            bk = _dedupe_bucket(toks)
            if bk:
                buckets.append(bk)
        return buckets

    # ---------- LIST/TUPLE ----------
    if isinstance(raw, (list, tuple)):
        raw_list = list(raw)
        if len(raw_list) == 0:
            return []

        # list[dict] -> each dict is a bucket
        if all(isinstance(g, dict) for g in raw_list):
            out: List[List[Any]] = []
            for bucket_obj in raw_list:
                items = [k for k, v in bucket_obj.items() if not _is_none_val(v)]
                bk = _dedupe_bucket(items)
                if bk:
                    out.append(bk)
            return out

        # list of iterables -> buckets
        if all(isinstance(g, (list, tuple, set)) for g in raw_list):
            out = []
            for g in raw_list:
                bk = _dedupe_bucket(list(g))
                if bk:
                    out.append(bk)
            return out

        # list of scalars -> single bucket
        if all(not isinstance(g, (list, tuple, set, dict)) for g in raw_list):
            b = _dedupe_bucket(raw_list)
            return [b] if b else []

        # mixed: coerce element-wise
        out = []
        for g in raw_list:
            if isinstance(g, dict):
                items = [k for k, v in g.items() if not _is_none_val(v)]
                bk = _dedupe_bucket(items)
            elif isinstance(g, (list, tuple, set)):
                bk = _dedupe_bucket(list(g))
            else:
                bk = _dedupe_bucket([g])
            if bk:
                out.append(bk)
        return out

    # ---------- DICT ----------
    if isinstance(raw, dict):
        for key in [
            "groups",
            "buckets",
            "sequence",
            "timeline",
            "temporal_groups",
            "ordered_groups",
            "result",
            "temporal_result",
            "output",
        ]:
            if key in raw:
                return _coerce_to_buckets(raw[key], normalize=normalize)

        # dict of symptom->mentions (single bucket)
        items = [k for k, v in raw.items() if not _is_none_val(v)]
        b = _dedupe_bucket(items)
        return [b] if b else []

    return []


# ---------------- Evaluation wrappers ----------------

def evaluate_temporal_sample(
    gold_raw: Any,
    pred_raw: Any,
    normalize: Optional[Callable[[Any], Any]] = _normalize_default,
    compute_union_position: bool = True,
    lccs_alignment: bool = True,
    combine_method: Optional[str] = "product",  # None | product | geom | weighted_geom
) -> Dict[str, Any]:
    """Evaluate one (gold, pred) pair."""
    gold = _coerce_to_buckets(gold_raw, normalize=normalize)
    pred = _coerce_to_buckets(pred_raw, normalize=normalize)

    em_ok, em_reason = strict_exact_match(gold, pred, normalize=normalize)

    item_P, item_R, item_F1, cov_details = item_set_prf1(gold, pred, normalize=normalize)

    tau_b, tau_details = kendall_tau_b_ties(gold, pred, normalize=normalize, return_details=True)

    pos_score_int, pos_mae_norm_int, pos_details_int = position_error_distance(
        gold, pred, normalize=normalize, missing_penalty="ignore", return_details=True
    )

    pos_score_union, pos_mae_norm_union, pos_details_union = None, None, None
    if compute_union_position:
        pos_score_union, pos_mae_norm_union, pos_details_union = position_error_distance(
            gold, pred, normalize=normalize, missing_penalty="max", return_details=True
        )

    lccs_len, lccs_rec_g, lccs_prec_p, lccs_f1, lccs_align = group_aware_lccs(
        gold, pred, normalize=normalize, return_alignment=lccs_alignment
    )

    combined = None
    if combine_method is not None:
        if combine_method == "product":
            combined = tau_b * item_F1
        elif combine_method == "geom":
            combined = (tau_b * item_F1) ** 0.5
        elif combine_method == "weighted_geom":
            w_tau, w_f1 = 0.6, 0.4
            denom = max(1e-9, w_tau + w_f1)
            combined = ((tau_b ** w_tau) * (item_F1 ** w_f1)) ** (1.0 / denom)
        else:
            raise ValueError("combine_method must be one of: None | 'product' | 'geom' | 'weighted_geom'")

    return {
        "exact_match": {"match": em_ok, "reason": em_reason},
        "coverage": {"precision": item_P, "recall": item_R, "f1": item_F1, "details": cov_details},
        "kendall_tau_b": {"tau_b": tau_b, "details": tau_details},
        "position_error_intersection": {"score": pos_score_int, "mae_norm": pos_mae_norm_int, "details": pos_details_int},
        "position_error_union": None
        if pos_score_union is None
        else {"score": pos_score_union, "mae_norm": pos_mae_norm_union, "details": pos_details_union},
        "lccs": {
            "length": lccs_len,
            "recall_gold": lccs_rec_g,
            "precision_pred": lccs_prec_p,
            "f1": lccs_f1,
            "alignment": lccs_align,
        },
        "combined_order_x_coverage": {"method": combine_method, "score": combined},
        "canonical_inputs": {"gold": gold, "pred": pred},
    }


def _mean_safe(vals: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def evaluate_temporal_batch(
    gold_list: List[Any],
    pred_list: List[Any],
    normalize: Optional[Callable[[Any], Any]] = _normalize_default,
    compute_union_position: bool = True,
    lccs_alignment: bool = False,
    combine_method: Optional[str] = "product",
) -> Dict[str, Any]:
    """Evaluate a batch of (gold, pred) pairs."""

    if len(gold_list) != len(pred_list):
        raise ValueError(f"gold_list and pred_list must have same length; got {len(gold_list)} vs {len(pred_list)}")

    per_sample: List[Dict[str, Any]] = []
    n = len(gold_list)

    micro_gold_total = 0
    micro_pred_total = 0
    micro_inter_total = 0

    ems: List[float] = []
    tau_bs: List[float] = []

    cov_Ps: List[float] = []
    cov_Rs: List[float] = []
    cov_F1s: List[float] = []

    pos_int_scores: List[float] = []
    pos_int_maes: List[float] = []

    pos_union_scores: List[float] = []
    pos_union_maes: List[float] = []

    lccs_lens: List[float] = []
    lccs_recalls: List[float] = []
    lccs_precs: List[float] = []
    lccs_f1s: List[float] = []

    combined_scores: List[float] = []

    failures = 0

    for idx, (g_raw, p_raw) in enumerate(zip(gold_list, pred_list)):
        try:
            res = evaluate_temporal_sample(
                g_raw,
                p_raw,
                normalize=normalize,
                compute_union_position=compute_union_position,
                lccs_alignment=lccs_alignment,
                combine_method=combine_method,
            )
            per_sample.append({"index": idx, **res})

            ems.append(1.0 if res["exact_match"]["match"] else 0.0)

            cov = res["coverage"]
            cov_Ps.append(cov["precision"])
            cov_Rs.append(cov["recall"])
            cov_F1s.append(cov["f1"])

            cd = cov["details"]
            micro_gold_total += cd["gold_count"]
            micro_pred_total += cd["pred_count"]
            micro_inter_total += cd["intersection"]

            tau_bs.append(res["kendall_tau_b"]["tau_b"])

            posi = res["position_error_intersection"]
            pos_int_scores.append(posi["score"])
            pos_int_maes.append(posi["mae_norm"])

            if res["position_error_union"] is not None:
                posu = res["position_error_union"]
                pos_union_scores.append(posu["score"])
                pos_union_maes.append(posu["mae_norm"])

            lccs = res["lccs"]
            lccs_lens.append(lccs["length"])
            lccs_recalls.append(lccs["recall_gold"])
            lccs_precs.append(lccs["precision_pred"])
            lccs_f1s.append(lccs["f1"])

            if res["combined_order_x_coverage"]["score"] is not None:
                combined_scores.append(res["combined_order_x_coverage"]["score"])

        except Exception as e:
            failures += 1
            per_sample.append({"index": idx, "error": str(e), "raw_gold": g_raw, "raw_pred": p_raw})

    evaluated = n - failures

    macro = {
        "exact_match_rate": _mean_safe(ems),
        "coverage_macro": {"precision": _mean_safe(cov_Ps), "recall": _mean_safe(cov_Rs), "f1": _mean_safe(cov_F1s)},
        "kendall_tau_b": _mean_safe(tau_bs),
        "position_error_intersection": {"score": _mean_safe(pos_int_scores), "mae_norm": _mean_safe(pos_int_maes)},
        "position_error_union": None
        if not pos_union_scores
        else {"score": _mean_safe(pos_union_scores), "mae_norm": _mean_safe(pos_union_maes)},
        "lccs": {"length": _mean_safe(lccs_lens), "recall_gold": _mean_safe(lccs_recalls), "precision_pred": _mean_safe(lccs_precs), "f1": _mean_safe(lccs_f1s)},
        "combined_order_x_coverage": _mean_safe(combined_scores) if combined_scores else None,
    }

    if micro_pred_total == 0 and micro_gold_total == 0:
        micro_cov = {"precision": 1.0, "recall": 1.0, "f1": 1.0, "totals": {"gold": 0, "pred": 0, "intersection": 0}}
    else:
        micro_P = (micro_inter_total / micro_pred_total) if micro_pred_total else 1.0
        micro_R = (micro_inter_total / micro_gold_total) if micro_gold_total else 1.0
        micro_F1 = (2 * micro_P * micro_R / (micro_P + micro_R)) if (micro_P + micro_R) else 0.0
        micro_cov = {
            "precision": micro_P,
            "recall": micro_R,
            "f1": micro_F1,
            "totals": {"gold": micro_gold_total, "pred": micro_pred_total, "intersection": micro_inter_total},
        }

    return {
        "n_samples": n,
        "n_evaluated": evaluated,
        "n_failed": failures,
        "macro_averages": macro,
        "micro_coverage": micro_cov,
        "per_sample": per_sample,
    }


# ---------------- Class API ----------------

class MedTempoEvaluator:
    """Callable evaluator for MedTEMPO-style bucketed timelines."""

    def __init__(
        self,
        normalize: Callable[[Any], Any] = _normalize_default,
        compute_union_position: bool = True,
        lccs_alignment: bool = False,
        combine_method: Optional[str] = "product",
    ) -> None:
        self.normalize = normalize
        self.compute_union_position = compute_union_position
        self.lccs_alignment = lccs_alignment
        self.combine_method = combine_method

    def evaluate_sample(self, gold: Any, pred: Any) -> Dict[str, Any]:
        return evaluate_temporal_sample(
            gold,
            pred,
            normalize=self.normalize,
            compute_union_position=self.compute_union_position,
            lccs_alignment=self.lccs_alignment,
            combine_method=self.combine_method,
        )

    def evaluate_batch(self, gold_list: List[Any], pred_list: List[Any]) -> Dict[str, Any]:
        return evaluate_temporal_batch(
            gold_list,
            pred_list,
            normalize=self.normalize,
            compute_union_position=self.compute_union_position,
            lccs_alignment=self.lccs_alignment,
            combine_method=self.combine_method,
        )

    def evaluate_from_dataframes(
        self,
        pred_df: "Any",
        gold_df: "Any",
        *,
        pred_col: str,
        gold_col: str,
        id_col: Optional[str] = None,
        how: str = "inner",
        sort_by_id: bool = True,
        return_per_sample: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate predictions and gold stored in pandas DataFrames.

        Args:
            pred_df: pandas DataFrame containing predictions.
            gold_df: pandas DataFrame containing gold.
            pred_col: column name in pred_df for predicted output.
            gold_col: column name in gold_df for gold output.
            id_col: if provided, align pred/gold by this id via merge.
            how: merge strategy when id_col is provided (default: inner).
            sort_by_id: if True and id_col is provided, sort merged rows by id.
            return_per_sample: if False, drop per_sample to keep output small.

        Returns:
            A dict like evaluate_batch(), optionally without per_sample.
        """
        import pandas as pd  # local import to keep module light if pandas isn't needed

        if not hasattr(pred_df, "__dataframe__") and not isinstance(pred_df, pd.DataFrame):
            pred_df = pd.DataFrame(pred_df)
        if not hasattr(gold_df, "__dataframe__") and not isinstance(gold_df, pd.DataFrame):
            gold_df = pd.DataFrame(gold_df)

        if id_col is None:
            if len(pred_df) != len(gold_df):
                raise ValueError(
                    f"When id_col is None, pred_df and gold_df must have the same length; got {len(pred_df)} vs {len(gold_df)}"
                )
            pred_list = pred_df[pred_col].tolist()
            gold_list = gold_df[gold_col].tolist()
        else:
            merged = pred_df[[id_col, pred_col]].merge(
                gold_df[[id_col, gold_col]],
                on=id_col,
                how=how,
                suffixes=("_pred", "_gold"),
            )
            if sort_by_id:
                merged = merged.sort_values(id_col)

            pred_list = merged[pred_col].tolist()
            gold_list = merged[gold_col].tolist()

        out = self.evaluate_batch(gold_list=gold_list, pred_list=pred_list)
        if not return_per_sample:
            out = dict(out)
            out.pop("per_sample", None)
        return out


__all__ = [
    "MedTempoEvaluator",
    "evaluate_temporal_sample",
    "evaluate_temporal_batch",
    "strict_exact_match",
    "group_aware_lccs",
    "kendall_tau_b_ties",
    "item_set_prf1",
    "position_error_distance",
]


if __name__ == "__main__":
    # Lightweight self-check / example usage
    gold = [["fever", "headache"], ["rash"], ["fatigue", "nausea"]]
    pred = [["headache", "fever"], ["cough"], ["nausea", "fatigue"]]

    evaluator = MedTempoEvaluator()
    res = evaluator.evaluate_sample(gold, pred)
    print("Exact match:", res["exact_match"]["match"], res["exact_match"]["reason"])
    print("Macro coverage F1:", res["coverage"]["f1"])
    print("Tau-b:", res["kendall_tau_b"]["tau_b"])


