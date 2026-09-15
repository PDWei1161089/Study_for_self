# p4_1.py
score = int(input("请输入一个分数："))

if 90 <= score <= 100:
    print("A")
if 80 <= score < 90:
    print("B")
if 60 <= score < 80:
    print("C")
if 0 <= score < 60:
    print("D")
if score < 0 or score > 100:
    print("输入错误！")