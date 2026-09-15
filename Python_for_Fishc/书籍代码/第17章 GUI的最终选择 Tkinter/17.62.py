# 17.62.py
from tkinter import *
from tkinter.filedialog import *

root = Tk()


def callback():
    fileName = askopenfilename()
    print(fileName)


Button(root, text="打开文件", command=callback).pack()

mainloop()
