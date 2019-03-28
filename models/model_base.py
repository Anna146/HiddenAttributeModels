import sys

sys.path.insert(0, "utils")
from config import paramstring_skip, always_params


class MODEL_BASE(object):
    MODELS = {}

    @classmethod
    def register(cls, modelcls):
        name = modelcls.__name__

        # ensure we don't have two modules containing model classes with the same name
        if name in cls.MODELS and cls.MODELS[name] != modelcls:
            raise RuntimeError("encountered two models with the same name: {%s}" % (name))

        cls.MODELS[name] = modelcls
        return modelcls

    @staticmethod
    def config():
        raise NotImplementedError("config method must be provided by subclass")

    @staticmethod
    def build():
        raise NotImplementedError("build method be provided by subclass")

    def params_to_string(self, params, skip_check=False):
        """ Convert params to a string """
        params = {k: self.param_types[k](str(v)) for k, v in params.items() if k not in paramstring_skip}
        s = [params["model"]]

        for k in sorted(params):
            # handled separately
            if k == "model":
                continue
            if k not in always_params and k in self.param_defaults and params[k] == self.param_defaults[k]:
                continue
            s.append("%s--%s" % (k, params[k]))

        s = "__".join(s)

        if not skip_check:
            if params != self.string_to_params(s, True):
                d = self.string_to_params(s, True)
                for k, v in d.items():
                    if k not in params or params.get(k) != v:
                        print("%s k=%s vs. k=%s" % (k, params.get(k), d[k]))
                        print(type(params.get(k)), type(d[k]))
                for k, v in params.items():
                    if k not in d or d.get(k) != v:
                        print("%s k=%s vs. k=%s" % (k, params[k], d.get(k)))
                        print(type(params[k]), type(d.get(k)))
                print("dict:", sorted(d.items()))
                print(" str:", s)
                raise RuntimeError("self.string_to_params(s, True)")
        return s

    def string_to_params(self, s, skip_check=False):
        """ Convert a param string back to a dict """
        fields = s.split("__")
        modelname = fields[0]
        params = fields[1:]

        out = {k: v for k, v in self.param_defaults.items() if k not in paramstring_skip}
        out["model"] = modelname

        for pstr in params:
            print(pstr)
            k, v = pstr.split("--")
            assert k not in paramstring_skip, "invalid key '%s' encountered in string: %s" % (k, s)
            out[k] = self.param_types[k](v)

        if not skip_check:
            assert s == self.params_to_string(out, True), "asymmetric string_to_params on string: %s" % s
        return out

    def __init__(self, p, param_defaults, param_types):
        self.p = p
        self.param_defaults = param_defaults
        self.param_types = param_types

    @staticmethod
    def save(outfn):
        raise NotImplementedError("should save model to outfn")

    def load(self, weight_file):
        raise NotImplementedError("should load model weights from weight file")
