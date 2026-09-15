# p6_4.py

def discount(price, rate):
    final_price = price * rate
    # 下面试图修改全局变量的值
    old_price = 50
    print(f"在局部变量中修改后old_price的值为：{old_price:.2f}")
    return final_price


old_price = float(input("请输入原价："))
rate = float(input("请输入折扣率："))
new_price =  discount(old_price, rate)

print(f"全局变量old_price现在的值是：{old_price:.2f}")
print(f"打折后价格是:{new_price:.2f}")
