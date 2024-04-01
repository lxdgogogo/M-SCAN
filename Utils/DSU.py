class DSU:
    def __init__(self, num: int):
        self.root = [i for i in range(num)]

    def find(self, k):
        if k >= len(self.root):
            print(k)
        if self.root[k] == k:
            return k
        self.root[k] = self.find(self.root[k])
        return self.root[k]

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        if x != y:
            self.root[y] = x
