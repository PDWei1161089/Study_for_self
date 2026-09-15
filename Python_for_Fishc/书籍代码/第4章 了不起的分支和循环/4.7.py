# p4_7.py
score = int(input("请输入一个分数："))

level = "A" if 90 <= score <= 100 else "B" if 80 <= score < 90 else "C" if 60 <= score < 80 else "D" if 0 <= score < 60 else print("输入错误！")

print(level)
