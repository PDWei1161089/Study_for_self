# 17.63.py
from tkinter import *
from tkinter.colorchooser import *

root = Tk()


def callback():
    fileName = askcolor()
    print(fileName)


Button(root, text="选择颜色", command=callback).pack()

mainloop()
