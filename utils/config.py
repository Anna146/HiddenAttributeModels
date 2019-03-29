import os

# params that should not appear in the param string (because they're part of the path or not relevant)
paramstring_skip = ["expname", "outdir", "train_years", "test_year", "outep"]

# params that should appear in the param string even when set to their default values
always_params = ["simdim", "maxqlen", "kernel_num", "k", "batch_size", "hidden_size", "hidden_size_attention", "attention_type"]

# type conversion functions to use for types where the default doesn't work (e.g., bool('false') is True)
forced_types = {type(True): lambda x: x.lower() == "true", type(None): lambda x: None if x.lower() == "none" else x}

# params shared by all models; model-specific params are set within each model class
def pipeline_config():
    expname = "default"  # experiment name
    model = "DoubleAtt"  # default model
    grids_dir = "grids"
    dump_dir = "dump"
    models_dir = "checkpoints"
    results_dir = "results"

    seed = 33
    mode = "grid"
    grid_file = "%s/%s_%s.txt" % (grids_dir, model, expname)
    train_folder = "data/reddit_data/train_profession"
    test_set = "data/reddit_data/test_profession.txt"
    results_file = ""

    # vocabulary
    weights_path = "data/reddit_data/weights.npy"

    return locals().copy()  # ignored by sacred

