# 9.5.py
try:
    int("abc")
    sum_ = 1 + "1"
    f = open("我是一个不存在的文档.txt")
    print(f.read())
    f.close()
except (ValueError, OSError, TypeError) as reason:
    print("出错啦T_T\n错误原因是: " + str(reason))