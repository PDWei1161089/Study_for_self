# 13.7.py
def c2f(cel):
    fah = cel * 1.8 + 32
    return fah


def f2c(fah):
    cel = (fah - 32) / 1.8
    return cel


def test():
    print(f"测试, 0摄氏度 = {c2f(0):.2f}华氏度")
    print(f"测试，0华氏度 = {f2c(0):.2f}摄氏度")


if __name__ == "__main__":
    test()
