# 17.2.py
import tkinter as tk


class App:
    def __init__(self, root):
        # 创建一个框架，然后在里边添加一个Button按钮组件
        # 框架一般是在复杂的布局中起到组件分组的作用
        frame = tk.Frame(root)
        frame.pack(padx=400, pady=300)  # LEFT、TOP(默认)、RIGHT、BOTTOM

        # 创建一个按钮组件，fg就是foreground的缩写，设置前景色的意思, bg是背景色的缩写
        self.hi_there = tk.Button(frame, text="打招呼", fg="white", bg="black", command=self.say_hi)
        self.hi_there.pack(side=tk.LEFT)

    def say_hi(self):
        print("互联网的广大朋友们大家好，我是小甲鱼!")


# 设置一个toplevel的根窗口，并把它作为参数实例化app对象
root = tk.Tk()
app = App(root)

# 开始主事件循环
root.mainloop()
