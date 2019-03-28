import importlib
import os.path
from glob import glob
import sys

sys.path.insert(0, "/home/tigunova/PycharmProjects/ham_rep")

# import all model modules so that the model classes are registered
pwd = os.path.dirname(__file__)
for fn in glob(os.path.join(pwd, "*.py")):
    modname = os.path.basename(fn)[:-3]
    if not modname.startswith("__"):
        importlib.import_module("models.%s" % (modname))
