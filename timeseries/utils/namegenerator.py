class NameGenerator(object):
    _VARIABLE_COUNT = 0

    def __init__(self, prefix='var_'):
        self._prefix = prefix

    def get(self):
        output = "{}{}".format(self._prefix, self._VARIABLE_COUNT)
        self._VARIABLE_COUNT += 1
        return output

    def reset(self):
        self._VARIABLE_COUNT = 0
