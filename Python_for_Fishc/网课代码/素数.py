#本脚本适用于查看确定范围内的素数集合
import time
print("此脚本适用于查看确定范围内的素数集合")
time.sleep(2)
list1 = []
condition = input("是否仅查看素数？")
time.sleep(0.5)
a = input("请输入范围的最小值：")
time.sleep(0.5)
b = input("请输入范围的最大值：")
numa = int(a)
numb = int(b)

while numa < 2 or numb < 2:
    print("请输入大于1的整数值！！！")
    time.sleep(0.5)
    a = input("请输入范围的最小值：")
    time.sleep(0.5)
    b = input("请输入范围的最大值：")
    time.sleep(1)
    numa = int(a)
    numb = int(b)
while numa > numb:
    print("请输入正确的范围！！！")
    time.sleep(1)
    a = input("请输入范围的最小值：")
    time.sleep(1)
    b = input("请输入范围的最大值：")
    numa = int(a)
    numb = int(b)

time.sleep(1)

for n in range(numa,numb + 1):
    for x in range(2,n):
        if n % x == 0:
            if condition == "否":
                print(n, "=" , x ,"*", n//x)
            break
    else:
        list1.append(n)

ans = str(list1)
numl = len(list1)
list2 = str(list1)
time.sleep(2)

if numl == 0:
    print('此范围内无素数。')
else:
    print(a +"至" +b + "内的素数集合为："+list2)

print("感谢您的配合，再见！")
