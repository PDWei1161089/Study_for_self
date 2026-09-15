# p4_2.py
score = int(input("请输入您的分数："))

if 90 <= score <= 100:
    print("A")
else:
    if 80 <= score < 90:
        print("B")
    else:
        if 60 <= score < 80:
            print("C")
        else:
            if 0 <= score < 60:
                print("D")
            else:
                print("输入错误！")