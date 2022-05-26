class Compressor(object):
    def __init__(self, w):
        self._w = w

    def compress(self, x):
        raise NotImplementedError

    @property
    def w(self):
        return self._w

    def __call__(self, x):
        return self.compress(x)


class Identity(Compressor):
    def __init__(self):
        super().__init__(1.)

    def compress(self, x):
        return x
