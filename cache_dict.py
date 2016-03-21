

class CacheDict:

    def __init__(self, max_len=50):
        self.c_dict = {}
        self.max_len = max_len
        assert max_len > 0

    def update(self, k, v):
        if k in self.c_dict:
            return
        if len(self.c_dict) >= self.max_len:
            self.c_dict = {}
        self.c_dict[k] = v

    def drop(self, k):
        if k not in self.c_dict:
            return

        self.c_dict.pop(k)

    def get(self, k):
        if k in self.c_dict:
            return self.c_dict[k]
        else:
            return None
