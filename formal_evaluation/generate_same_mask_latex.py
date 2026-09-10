#!/usr/bin/env python3
"""Generate four protocol-matched LaTeX reports from a completed same-mask summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SCHEMES = {
    "unfiltered": ("Unfiltered", "same_mask_v7_unfiltered.tex"),
    "p97_5": ("Joint8 P97.5", "same_mask_v7_joint8_p97_5.tex"),
    "all8_p95": ("Joint8 P95", "same_mask_v7_joint8_p95.tex"),
    "temporal_p95_other_p97_5": ("Mixed", "same_mask_v7_mixed.tex"),
}
DATASETS = ("h2o", "hot3d", "arctic", "oakink_v2", "taco", "hoi4d")
DATASET_LABELS = {
    "h2o": "H2O", "hot3d": "HOT3D", "arctic": "ARCTIC",
    "oakink_v2": "OakInk-v2", "taco": "TACO", "hoi4d": "HOI4D",
}
METHODS = (
    "egofound3r_stride5", "egofound3r", "wilor", "hawor", "pad_hand",
    "reviv4d", "s2contact", "contactopt", "vggt", "pi3", "da3_large_1_1",
    "lingbot_map_long", "vggt_omega",
)
METHOD_LABELS = {
    "egofound3r_stride5": "EgoFound3R stride5",
    "egofound3r": "EgoFound3R (legacy)",
    "wilor": "WiLoR", "hawor": "HaWoR", "pad_hand": "PAD-Hand",
    "reviv4d": "ReViV4D", "s2contact": r"S$^2$Contact",
    "contactopt": "ContactOpt", "vggt": "VGGT", "pi3": "Pi3",
    "da3_large_1_1": "DA3-Large-1.1",
    "lingbot_map_long": "LingBot-Map-Long", "vggt_omega": "VGGT-Omega",
    "dyn_hamr": r"Dyn-HaMR$^{100\mathrm{w}}$",
    "interactvlm": r"InteractVLM$^{100\mathrm{w}}$",
}
HAND_METHODS = ("egofound3r_stride5", "egofound3r", "wilor", "hawor", "pad_hand", "reviv4d")
HAND_DISPLAY_METHODS = HAND_METHODS + ("dyn_hamr",)
SCENE_METHODS = (
    "egofound3r_stride5", "egofound3r", "reviv4d", "vggt", "pi3",
    "da3_large_1_1", "lingbot_map_long", "vggt_omega",
)
CONTACT_METHODS = ("egofound3r_stride5", "egofound3r", "s2contact", "contactopt")
CONTACT_DISPLAY_METHODS = CONTACT_METHODS + ("interactvlm",)

SCENE_METRICS = (
    ("ATE (m)", "camera_ate_aligned", False, 4),
    (r"Rot ($^\circ$)", "camera_rot_error_deg", False, 2),
    ("AbsRel", "depth_abs_rel", False, 3),
    ("RMSE (m)", "depth_rmse", False, 3),
    (r"$\delta_1", "depth_delta1", True, 3),
    ("AUC@30", "camera_pose_auc_30", True, 3),
)
CONTACT_METRICS = (
    ("Joint P", "joint_contact_precision", True, 3),
    ("Joint R", "joint_contact_recall", True, 3),
    ("Joint F1", "joint_contact_f1", True, 3),
    ("Marker P", "marker_contact_precision", True, 3),
    ("Marker R", "marker_contact_recall", True, 3),
    ("Marker F1", "marker_contact_f1", True, 3),
    ("Joint Vis P", "joint_visibility_precision", True, 3),
    ("Joint Vis R", "joint_visibility_recall", True, 3),
    ("Joint Vis F1", "joint_visibility_f1", True, 3),
    ("Marker Vis P", "marker_visibility_precision", True, 3),
    ("Marker Vis R", "marker_visibility_recall", True, 3),
    ("Marker Vis F1", "marker_visibility_f1", True, 3),
)
HAND = {
    "joint": (
        "Hand metrics: 21 joints",
        ("MPJPE", "RR", "PA", "Sim3", "W", "WA", "MPJVE", "MPJAE"),
        ("mpjpe", "rr_mpjpe", "pa_mpjpe", "global_sim3_mpjpe", "w_mpjpe",
         "wa_mpjpe", "mpjve", "mpjae"),
    ),
    "marker": (
        "Hand metrics: 195 markers",
        ("MPMPE", "RR", "PA", "Sim3", "W", "WA", "MPMVE", "MPMAE"),
        ("marker_mpmpe", "marker_rr_mpmpe", "marker_pa_mpmpe",
         "marker_global_sim3_mpmpe", "marker_w_mpmpe", "marker_wa_mpmpe",
         "marker_mpmve", "marker_mpmae"),
    ),
    "vertex": (
        "Hand metrics: 778 vertices",
        ("MPVPE", "RR", "PA", "Sim3", "W", "WA", "MPVVE", "MPVAE"),
        ("vertex_mpvpe", "vertex_rr_mpvpe", "vertex_pa_mpvpe",
         "vertex_global_sim3_mpvpe", "vertex_w_mpvpe", "vertex_wa_mpvpe",
         "vertex_mpvve", "vertex_mpvae"),
    ),
}


def completed_summary(state_path: Path) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs = list(state.get("jobs", {}).values())
    if len(jobs) != 1 or jobs[0].get("status") != "done":
        raise ValueError("expected one completed task job")
    summary = jobs[0].get("completion_evidence", {}).get("summary.json")
    if not isinstance(summary, dict) or summary.get("status") != "complete":
        raise ValueError("completed summary.json is unavailable in task state")
    if tuple(sorted(summary.get("schemes", {}))) != tuple(sorted(SCHEMES)):
        raise ValueError("unexpected scheme set")
    return summary


def report_rows(state_path: Path, method: str) -> dict[str, dict]:
    """Read exact report metrics already captured by a registered task inspect."""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    for job in state.get("jobs", {}).values():
        reports = job.get("report_summary", {}).get("reports", {})
        for report in reports.values():
            for dataset, payload in report.get("methods", {}).get(method, {}).items():
                metrics = payload.get("metrics")
                if not isinstance(metrics, dict):
                    continue
                if dataset in rows and rows[dataset] != metrics:
                    raise ValueError(f"conflicting {method} metrics for {dataset}")
                rows[dataset] = metrics
    return rows


def merge_external_rows(summary: dict, dyn_states: list[Path], interact_state: Path | None,
                        visibility_summary: Path | None) -> dict:
    merged = json.loads(json.dumps(summary))
    dyn_rows: dict[str, dict] = {}
    for state in dyn_states:
        for dataset, row in report_rows(state, "dyn_hamr").items():
            if dataset in dyn_rows and dyn_rows[dataset] != row:
                raise ValueError(f"conflicting dyn_hamr metrics for {dataset}")
            dyn_rows[dataset] = row
    interact_rows = report_rows(interact_state, "interactvlm") if interact_state else {}
    if dyn_states and set(dyn_rows) != set(DATASETS):
        raise ValueError(f"Dyn-HaMR datasets mismatch: {sorted(dyn_rows)}")
    if interact_state and set(interact_rows) != set(DATASETS):
        raise ValueError(f"InteractVLM datasets mismatch: {sorted(interact_rows)}")
    for scheme_data in merged["schemes"].values():
        for dataset in DATASETS:
            if dataset in dyn_rows:
                scheme_data[dataset]["dyn_hamr"] = dyn_rows[dataset]
            if dataset in interact_rows:
                scheme_data[dataset]["interactvlm"] = interact_rows[dataset]
    if visibility_summary:
        visibility = json.loads(visibility_summary.read_text(encoding="utf-8"))
        if set(visibility.get("schemes", {})) != set(SCHEMES):
            raise ValueError("visibility summary scheme set mismatch")
        for scheme, scheme_data in visibility["schemes"].items():
            for dataset, methods in scheme_data.items():
                for method, metrics in methods.items():
                    merged["schemes"][scheme][dataset].setdefault(method, {}).update(metrics)
    return merged


def finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def combined_hand(row: dict, stem: str) -> float | None:
    weighted = 0.0
    count = 0
    for side in ("left", "right"):
        prefix = f"hand_{side}_{stem}"
        mean = finite(row.get(prefix + "_mean"))
        side_count = int(row.get(prefix + "_count") or 0)
        if mean is not None and side_count:
            weighted += mean * side_count
            count += side_count
    if not count:
        return None
    value = weighted / count
    return value / 1000.0 if stem.endswith("ae") else value


def scalar(row: dict, stem: str) -> float | None:
    return finite(row.get(stem + "_mean"))


def best_methods(values: dict[str, float | None], higher: bool) -> set[str]:
    available = {method: value for method, value in values.items() if value is not None}
    if not available:
        return set()
    target = (max if higher else min)(available.values())
    return {method for method, value in available.items() if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)}


def formatted(value: float | None, decimals: int, bold: bool) -> str:
    if value is None:
        return "--"
    result = f"{value:.{decimals}f}"
    return rf"\textbf{{{result}}}" if bold else result


def table(title: str, label: str, metric_specs: tuple, methods: tuple[str, ...], data: dict,
          extractor, *, ranked_methods: tuple[str, ...] | None = None,
          always_show_methods: tuple[str, ...] = ()) -> str:
    ranked_methods = methods if ranked_methods is None else ranked_methods
    columns = "ll" + "r" * len(metric_specs)
    headings = []
    for name, _, higher, _ in metric_specs:
        headings.append(name + (r"$\uparrow$" if higher else r"$\downarrow$"))
    lines = [
        r"\begin{landscape}", r"\begin{scriptsize}", rf"\begin{{longtable}}{{{columns}}}",
        rf"\caption{{{title}}}\label{{{label}}}\\", r"\toprule",
        "Dataset & Method & " + " & ".join(headings) + r" \\", r"\midrule", r"\endfirsthead",
        rf"\multicolumn{{{len(metric_specs) + 2}}}{{c}}{{\tablename\ \thetable\ -- continued}}\\",
        r"\toprule", "Dataset & Method & " + " & ".join(headings) + r" \\",
        r"\midrule", r"\endhead", r"\midrule",
        rf"\multicolumn{{{len(metric_specs) + 2}}}{{r}}{{Continued on next page}}\\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        rows = data[dataset]
        visible = [method for method in methods if method in rows and (
            method in always_show_methods or any(
                extractor(rows[method], stem) is not None for _, stem, _, _ in metric_specs
            )
        )]
        winners = []
        for _, stem, higher, _ in metric_specs:
            winners.append(best_methods(
                {
                    method: extractor(rows[method], stem)
                    for method in visible if method in ranked_methods
                },
                higher,
            ))
        for method in visible:
            cells = []
            for index, (_, stem, _, decimals) in enumerate(metric_specs):
                value = extractor(rows[method], stem)
                cells.append(formatted(value, decimals, method in winners[index]))
            lines.append(
                f"{DATASET_LABELS[dataset]} & {METHOD_LABELS[method]} & "
                + " & ".join(cells) + r" \\"
            )
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\end{longtable}", r"\end{scriptsize}", r"\end{landscape}", ""))
    return "\n".join(lines)


def hand_metric_specs(granularity: str) -> tuple:
    _, headings, stems = HAND[granularity]
    return tuple((heading, stem, False, 2) for heading, stem in zip(headings, stems, strict=True))


def strict_stride5_wins(data: dict) -> dict[str, list[str]]:
    result = {granularity: [] for granularity in HAND}
    for granularity, (_, headings, stems) in HAND.items():
        for dataset in DATASETS:
            rows = data[dataset]
            stride5 = rows["egofound3r_stride5"]
            for heading, stem in zip(headings, stems, strict=True):
                value = combined_hand(stride5, stem)
                baselines = [
                    combined_hand(rows[method], stem)
                    for method in HAND_METHODS if method != "egofound3r_stride5" and method in rows
                ]
                baselines = [item for item in baselines if item is not None]
                if value is not None and baselines and value < min(baselines):
                    result[granularity].append(f"{DATASET_LABELS[dataset]} x {heading}")
    return result


def render(summary: dict, scheme: str) -> tuple[str, dict[str, list[str]]]:
    title, _ = SCHEMES[scheme]
    data = summary["schemes"][scheme]
    wins = strict_stride5_wins(data)
    total = sum(map(len, wins.values()))
    protocol = (
        "No frames are excluded; every displayed method uses the same complete 2,378-window set."
        if scheme == "unfiltered" else
        f"Every displayed method uses the same Result-3 joint-derived {title} frame mask on the same "
        "2,378 windows. Alignments and per-frame residuals were frozen on each complete unfiltered "
        "window; masking only changes aggregation, and temporal metrics never bridge excluded frames."
    )
    parts = [
        r"\documentclass[10pt]{article}", r"\usepackage[margin=1.2cm]{geometry}",
        r"\usepackage{booktabs}", r"\usepackage{longtable}", r"\usepackage{pdflscape}",
        r"\usepackage[T1]{fontenc}", rf"\title{{Same-mask evaluation: {title}}}",
        r"\author{}", r"\date{}", r"\begin{document}", r"\maketitle",
        rf"\paragraph{{Protocol.}} {protocol}",
        r"\paragraph{Comparison.} Bold marks the best raw value among methods evaluated on the same 2,378 windows. Dyn-HaMR and InteractVLM carry the superscript $100\mathrm{w}$: their original 100-window values are repeated unchanged in all four documents, are not filtered, and are excluded from bolding and stride5 win counts. Missing native metrics are shown as --. The stride5 win count uses a stricter rule: its raw value must be strictly lower than every available same-window hand baseline value. Hand position errors are mm, velocity errors are mm/s, and acceleration errors are m/s$^2$.",
        rf"\paragraph{{Stride5 strict wins.}} {total}/144: "
        + "; ".join(f"{name.capitalize()} {len(items)}/48" for name, items in wins.items()) + ".",
        "",
        table("Scene / 3R metrics", f"tab:scene-{scheme}", SCENE_METRICS, SCENE_METHODS, data, scalar),
    ]
    for granularity, (caption, _, _) in HAND.items():
        parts.append(table(caption, f"tab:hand-{granularity}-{scheme}", hand_metric_specs(granularity),
                           HAND_DISPLAY_METHODS, data, combined_hand,
                           ranked_methods=HAND_METHODS, always_show_methods=("dyn_hamr",)))
    parts.append(table("Contact metrics", f"tab:contact-{scheme}", CONTACT_METRICS,
                       CONTACT_DISPLAY_METHODS, data, scalar,
                       ranked_methods=CONTACT_METHODS, always_show_methods=("interactvlm",)))
    parts.extend((r"\end{document}", ""))
    return "\n".join(parts), wins


def run(state: Path, output_dir: Path, dyn_states: list[Path],
        interact_state: Path | None, visibility_summary: Path | None) -> dict:
    summary = merge_external_rows(
        completed_summary(state), dyn_states, interact_state, visibility_summary
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_wins = {}
    for scheme, (_, filename) in SCHEMES.items():
        document, wins = render(summary, scheme)
        (output_dir / filename).write_text(document, encoding="utf-8")
        all_wins[scheme] = {
            "total": sum(map(len, wins.values())),
            "by_granularity": wins,
        }
    payload = {
        "source_run_id": "metrics-six-baseline12-plus-result3-directoss-same-mask-frozen-block16-p975-p95-mixed-20260909-v7-61cb6bb66a89",
        "definition": "strictly lower than every available same-dataset same-scheme hand baseline",
        "wins": all_wins,
    }
    (output_dir / "stride5_wins.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def self_check() -> None:
    row = {
        "hand_left_mpjpe_mean": 10.0, "hand_left_mpjpe_count": 2,
        "hand_right_mpjpe_mean": 20.0, "hand_right_mpjpe_count": 1,
        "hand_left_mpjae_mean": 2000.0, "hand_left_mpjae_count": 1,
    }
    assert math.isclose(combined_hand(row, "mpjpe"), 40 / 3)
    assert combined_hand(row, "mpjae") == 2.0
    assert best_methods({"a": 1.0, "b": 2.0}, False) == {"a"}
    print(json.dumps({"status": "self_check_passed"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dyn-state", action="append", type=Path, default=[])
    parser.add_argument("--interact-state", type=Path)
    parser.add_argument("--visibility-summary", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.state or not args.output_dir:
        parser.error("--state and --output-dir are required")
    print(json.dumps(run(
        args.state, args.output_dir, args.dyn_state,
        args.interact_state, args.visibility_summary,
    ), indent=2))


if __name__ == "__main__":
    main()
