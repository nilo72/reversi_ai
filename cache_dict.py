class CacheDict:

    def __init__(self):
        self.c_list = []
        self.flip = True

    def update(self, k, v):
        if len(self.c_list) < 2:
            self.c_list.append((k, v))
        else:
            if self.flip:
                self.c_list[0] = (k, v)
            else:
                self.c_list[1] = (k, v)
            self.flip = not self.flip

    def get(self, k):
        size = len(self.c_list)
        if size >= 1 and self.c_list[0][0] == k:
            return self.c_list[0][1]
        elif size >= 2 and self.c_list[1][0] == k:
            return self.c_list[1][1]
        else:
            return None
            
