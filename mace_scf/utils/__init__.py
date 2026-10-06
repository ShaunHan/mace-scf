from .train import train

from .script_utils import create_error_table

def is_fixed_point_model(model):
    module = model.module if hasattr(model, "module") else model
    return module.__class__.__name__ in ("FixedPoint", "FixedPointCore")


def create_scf_convergence_summary(*args, **kwargs):
    try:
        from .diagnostics import create_scf_convergence_summary as summarize
    except ModuleNotFoundError as exc:
        if exc.name != "mace_scf.utils.diagnostics":
            raise
        return "Optional diagnostics.py is not installed."
    return summarize(*args, **kwargs)


from .extend_arg_parse import extended_arg_parser

from .model_training_wrappers import make_model_wrapper
