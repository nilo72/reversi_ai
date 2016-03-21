from collections import deque
class CacheDict:

    def __init__(self, max_len=5):
        self.c_q = deque([])
        self.c_dict = {}
        self.max_len = max_len
        assert max_len > 0

    def update(self, k, v):
        if k in self.c_dict:
            return
        if len(self.c_q) >= self.max_len:
            # evict oldest item in cache
            evicted = self.c_q.popleft()
            self.c_dict.pop(evicted)
        self.c_q.append(k)
        self.c_dict[k] = v

    def drop(self, k):
        if k not in self.c_dict:
            return

        self.c_dict.pop(k)
        self.c_q.remove(k)

    def get(self, k):
        if k in self.c_dict:
            return self.c_dict[k]
        else:
            return None
