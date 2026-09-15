def max_(x):
    a = x[0]
    for i in x:
        if a > i:
            a = a
        else:
            a = i

    return a


print(max_(eval(input(""))))