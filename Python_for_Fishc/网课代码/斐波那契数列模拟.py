#迭代法（速度快）
def fibIter(n):   
        a = 1                           #赋值第一个数a 为 1
        b = 1                           #赋值第二个数b 为 1
        c = 1                           #赋值第三个数c 为 1
        while n > 2:                    #当所需代数为n， 且n 大于等于2的时候，开始循环
                c = a + b               #c是最终要输出的值，必须先进行这一步，就是要把这次循环的最终值弄出来。
                a = b                   #这个时候已经是新循环的第n - 2 项了(Thinking)
                b = c                   #这个时候是新循环的第 n - 1项(Thinking)
                n -= 1                  #进行下一次循环，直至到最后一次循环结束。
        return c                        #返回到最后一次循环的输出值


#递归法（速度慢，浪费资源）
def fibRecur(n):
        if n == 1 or n == 2:       #不符合条件的代数值1和2
                return 1                #直接返回1
        else:                           #其他情况
                return fibRecur(n - 1) + fibRecur(n - 2)                #斐波那契数列通项公式。

