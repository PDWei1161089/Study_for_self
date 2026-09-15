# 17.23.py
from tkinter import *

root = Tk()

text = Text(root, width=30, height=2)
text.pack()

# INSERT索引表示插入光标当前的位置
text.insert(INSERT, "I love\n")
text.insert(END, "FishC.com! ")

mainloop()