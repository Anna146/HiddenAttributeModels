"""
To run:
python run.py with hidden_size=$HS batch_size=$BS model=MLP train_set="data/ff_test.txt"
"""

import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")


sys.path.insert(0, "utils")
from init import sacred_init

ex, model_cls, param_defaults, param_types = sacred_init("run")


@ex.main
def main(_config, _run):
    p = _config

    print("train.py received params:")
    print(p)

    max_folds = 2

    model = model_cls(p, param_defaults, param_types)
    expid = model.params_to_string(p)
    print("param string: %s" % (expid))

    if p["mode"] == "grid":
        num_folds = 10
        with open(p["grid_file"], "a+") as f_grid:
            f_grid.write("params %s\n" % (expid))
            for kk in range(num_folds):
                if kk >= max_folds:
                    break
                print("Fold number " + str(kk))
                f_grid.write("Fold number " + str(kk) + "\n")
                model.train(p["train_folder"], kk=kk, folds=num_folds, grid=True, grid_file=f_grid)
    if p["mode"] == "test":
        model.train(p["train_folder"], grid=False)
        model.validate(p["test_set"])


if __name__ == "__main__":
    ex.run_commandline()
