# p4_6.py
score = int(input("请输入一个分数"))

if 90 <= score <= 100:
    level = "A"
elif 80 <= score < 90:
    level = "B"
elif 60 <= score < 80:
    level = "C"
elif 0 <= score <= 60:
    level = "D"
else:
    print("输入错误！")

print(level)
