x = 1
y = 1

while x < 10 and y < 10:
    while y <= x:
        answer = x * y
        print(x,"*",y,"=",answer, end = "      ")
        y += 1
    else:
        print()
        y = 1
        x += 1
    
        

    
