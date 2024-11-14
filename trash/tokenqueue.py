
class TokenQueue:

    def __init__(self):
        self.queue0 = []
        self.queue1 = []
        self.queue2 = []
        self.queue3 = []

    def put(self, item):
        if isinstance(item[0], str) and item[0] in self.queue0:
            item_index = self.queue0.index(item[0])
            origin_token1 = self.queue1[item_index]
            origin_token2  = item[1]
            self.queue1[item_index] = item[1]
            self.queue2[item_index] = max(self.queue2[item_index], item[2])
            self.queue3[item_index] += item[3]
            if isinstance(item[1], str) and origin_token1 != origin_token2:
                return (origin_token1, origin_token2, self.queue3[item_index])
        else:
            self.queue0.append(item[0])
            self.queue1.append(item[1])
            self.queue2.append(item[2])
            self.queue3.append(item[3])
        return None

    def get(self):
        if len(self.queue0) == 0:
            return None
        return (self.queue0.pop(0), self.queue1.pop(0), self.queue2.pop(0), self.queue3.pop(0))
    
    def is_empty(self):
        return len(self.queue0) == 0