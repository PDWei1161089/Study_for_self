# 9.8.py
def showMaxFactor(num):
    count = num // 2
    while count > 1:
        if num % count == 0:
            print(f"{num}的最大的约数是{count}")
            break
        count -= 1
    else:
        print(f"{num}是素数！")


num_ = int(input("请输入一个数: "))
showMaxFactor(num_)