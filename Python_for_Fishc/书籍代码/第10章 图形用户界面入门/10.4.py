# 10.4.py
import easygui as eg

file = open(r"..\第8章 永久存储\record2.txt")
eg.textbox(msg="文件【record.txt】的内容如下: ", title=" ", text=file.read())
