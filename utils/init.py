import importlib
import os
import sys
import sacred

from models.model_base import MODEL_BASE
from config import pipeline_config, forced_types


# hack to allow us to return locals() at the end of config functions
orig_dfb = sacred.config.config_scope.dedent_function_body


def _custom_config_dfb(*args, **kwargs):
    config_skip_return = "return locals().copy()  # ignored by sacred"
    src = orig_dfb(*args, **kwargs)
    filtered = [line for line in src.split("\n") if not line.strip() == config_skip_return]
    return "\n".join(filtered)


sacred.config.config_scope.dedent_function_body = _custom_config_dfb


def _load_model(name):
    if name not in MODEL_BASE.MODELS:
        raise RuntimeError("unknown model: {%s}; known models: {%s}" % (name, str(sorted(MODEL_BASE.MODELS))))
    model_cls = MODEL_BASE.MODELS[name]
    return model_cls


def sacred_init(name):
    ex = sacred.Experiment(name)
    ex.path = name
    sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("CUDA_VISIBLE_DEVICES")
    sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("USER")
    ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

    # manually parse args to find model name so we can run the model-specific config function
    pipeline_defaults = ex.config(pipeline_config)()
    mname = pipeline_defaults["model"]
    for arg in sys.argv:
        if arg.startswith("model="):
            mname = arg[len("model=") :]

    model_cls = _load_model(mname)
    model_defaults = ex.config(model_cls.config)()
    defaults = pipeline_defaults.copy()
    for k, v in model_defaults.items():
        assert k not in defaults, "pipeline params overlap with model params: {%s}" % (k)
        defaults[k] = v
    types = {k: forced_types.get(type(v), type(v)) for k, v in defaults.items()}

    ex.config(pipeline_config)
    ex.config(model_cls.config)

    return ex, model_cls, defaults, types
