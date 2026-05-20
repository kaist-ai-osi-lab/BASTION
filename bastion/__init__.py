"""Public API for the BASTION reproduction package."""

__all__ = [
    "build_adaptive_best_tree_from_draft_logits",
    "bastion_generate",
    "collect_calibration_data_from_jsonl",
    "fit_linear_calibration",
    "load_and_process_dataset",
]


def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name == "bastion_generate":
        from .tree_draft import bastion_generate

        return bastion_generate

    if name == "build_adaptive_best_tree_from_draft_logits":
        from .tree_draft import build_adaptive_best_tree_from_draft_logits

        return build_adaptive_best_tree_from_draft_logits

    if name == "collect_calibration_data_from_jsonl":
        from .cost_model import collect_calibration_data_from_jsonl

        return collect_calibration_data_from_jsonl

    if name == "fit_linear_calibration":
        from .cost_model import fit_linear_calibration

        return fit_linear_calibration

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
