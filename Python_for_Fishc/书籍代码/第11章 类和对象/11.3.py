# 11.3.py
class Turtle:
    def __init__(self, x):
        self.num = x


class Fish:
    def __init__(self, x):
        self.num = x


class Pool:
    def __init__(self, x, y):
        self.turtle = Turtle(x)
        self.fish = Fish(y)

    def print_num(self):
        print(f"水池里总共有乌龟{self.turtle.num}只，小鱼{self.fish.num}条")
