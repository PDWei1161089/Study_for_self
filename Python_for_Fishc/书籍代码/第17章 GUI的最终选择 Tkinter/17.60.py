# 17.60.py
from tkinter import *

root = Tk()


def callback():
    print("正中靶心")


photo = PhotoImage(file=r".\image\18.gif")
Label(root, image=photo).pack()

Button(root, text="点我", command=callback).place(relx=0.5, rely=0.5, anchor=CENTER)

mainloop()
