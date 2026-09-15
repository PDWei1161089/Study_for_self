# p6_3.py

def discount(price, rate):
    final_price = price * rate
    print(f"试图在函数内部访问全局变量old_price的值：{old_price:.2f}")
    return final_price


old_price = float(input("请输入原价："))
rate = float(input("请输入折扣率："))
new_price = discount(old_price, rate)

print(f"打折后价格是：{new_price:.2f}")
