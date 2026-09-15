# p6_2.py

def discount(price, rate):
    final_price = price * rate
    return final_price


old_price = float(input("请输入原价："))
rate = float(input("请输入折扣率："))
new_price = discount(old_price, rate)

print("打折后价格为：", new_price)
print(f"试图在函数外部访问变量final_price的值：{final_price:.2f}")
