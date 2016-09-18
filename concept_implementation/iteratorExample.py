#using iterator protocl
class Counter1:
    def __init__(self, low, high):
        self.current = low
        self.high = high

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

#using generator
def counter2(low, high):
    current = low
    while current <= high:
        yield current
        current += 1



for c in counter2(3, 8):
    print c