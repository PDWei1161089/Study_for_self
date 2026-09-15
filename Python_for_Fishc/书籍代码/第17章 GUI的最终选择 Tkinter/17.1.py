# 17.1.py
import tkinter as tk

# 创建一个主窗口，用于容纳整个GUI程序
root = tk.Tk()
# 设置主窗口对象的标题栏
root.title("FishC Demo")

# 添加一个Label组件，Label组件是GUI程序中最常用的组件之一
# Label组件可以显示文本、图标或者图片
# 在这里让它显示指定文本
theLabel = tk.Label(root, text="我的第二个窗口程序！")
# 然后调用Lable组件的pack()方法，用于自动调节组件自身的尺寸
theLabel.pack()

# 注意，这时候窗口还是不会显示的，除非执行下面这句代码
root.mainloop()
