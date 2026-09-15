# Euclidean algorithm 辗转相除法
def sort_ab(a, b):
    if a >= b:
        return a, b
    else:
        return b, a


def check_valid(temp_ab):
    flag_num = True

    while flag_num:
        try:
            a, b = temp_ab.split()
            int(a)
            int(b)
        except ValueError:
            print("您输入的数字格式有误，请按照如下条件重新输入: ")
            temp_ab = input("请输入两个正整数, 中间用空格隔开: ")
            continue
        if int(a) <= 0 or int(b) <= 0:
            print("您输入的数字格式有误，请按照如下条件重新输入: ")
            temp_ab = input("请输入两个正整数, 中间用空格隔开: ")
            continue

        print("==================================")
        print("正在进行下一步操作")
        print("==================================")
        return int(a), int(b)


def get_gcd(a, b, mode=0):
    a, b = sort_ab(a, b)  # 默认 a > b

    if a % b == 0:
        return b
    else:
        q = a // b
        r = a - b * q
        equation.append([a, q, b, r])

        if mode == 1:
            if q >= 0:
                print(f"{a} = {b} * {q} + {r}")
            else:
                print(f"{a} = {b} * (-{abs(q)}) + {r}")

        remainder.append([r, a, q, b])

        return get_gcd(b, r, mode)


def solve(remainder_li, gcd_ans, mode=0):
    r, a, q, b = remainder_li[0][0], remainder_li[0][1], remainder_li[0][2], remainder_li[0][3]
    coefficient_r = [[1, -q]]  # [co_r1], co_r1 = [co_a, co_b] = [A1, B1]
    equation_num = len(remainder_li)  # 式子个数

    if mode == 1:
        print("辗转相乘法过程如下:")

    if equation_num == 1:
        if mode == 1:
            if q <= 0:
                print(f"{r} = 1 * {a} - {abs(q)} * {b}")
            else:
                print(f"{r} = 1 * {a} + {q} * {b}")

        print("最终等式如下:")

        if q <= 0:
            return f"{a} - {abs(q)} * {b} = {r}"
        else:
            return f"{a} + {abs(q)} * {b} = {r}"

    elif equation_num == 2:
        if mode == 1:
            if q <= 0:
                print(f"{r} = 1 * {a} - {abs(q)} * {b}")
            else:
                print(f"{r} = 1 * {a} + {q} * {b}")

        r2, q2 = remainder_li[1][0], remainder_li[1][2]
        a2 = -q2 * coefficient_r[0][0]
        b2 = 1 - q2 * coefficient_r[0][1]

        if mode == 1:
            if b2 <= 0:
                print(f"{r2} = {a2} * {a} - {abs(b2)} * {b}")
            else:
                print(f"{r2} = {a2} * {a} + {b2} * {b}")

        print("最终等式如下:")
        if a2 < 0:
            return f"{a2} * {a} + {b2} * {b} = {r2}"
        elif b2 <= 0:
            return f"{a2} * {a} - {abs(b2)} * {b} = {r2}"

    else:
        if mode == 1:
            if q <= 0:
                print(f"{r} = 1 * {a} - {abs(q)} * {b}")
            else:
                print(f"{r} = 1 * {a} + {q} * {b}")

        r2, q2 = remainder_li[1][0], remainder_li[1][2]
        a2 = -q2 * coefficient_r[0][0]
        b2 = 1 - q2 * coefficient_r[0][1]
        coefficient_r.append([a2, b2])
        a1, b1 = coefficient_r[0][0], coefficient_r[0][1]
        a2, b2 = coefficient_r[1][0], coefficient_r[1][1]

        if mode == 1:
            if a2 <= 0:
                print(f"{r2} = ({a2}) * {a} + {b2} * {b}")
            elif b2 <= 0:
                print(f"{r2} = {a2} * {a} - {abs(b2)} * {b}")

        for i in range(3, equation_num+1):
            q3 = remainder_li[i-1][2]
            a3 = a1 - q3 * a2
            b3 = b1 - q3 * b2
            a1, b1 = a2, b2
            a2, b2 = a3, b3
            r3 = remainder_li[i-1][0]

            if mode == 1:
                if a3 <= 0:
                    print(f"{r3} = ({a3}) * {a} + {b3} * {b}")
                elif b3 <= 0:
                    print(f"{r3} = {a3} * {a} - {abs(b3)} * {b}")

        print("==================================")
        print("最终等式如下:")

        if a2 < 0:
            return f"{a2} * {a} + {b2} * {b} = {gcd_ans}"
        elif b2 <= 0:
            return f"{a2} * {a} - {abs(b2)} * {b} = {gcd_ans}"


def main():
    global equation, remainder, coefficient

    temp_ab = input("请输入两个正整数, 中间用空格隔开: ")
    a, b = check_valid(temp_ab)

    coefficient = [0, 0]  # a, b前面的系数
    interact1 = input("是否显示求最大公因数的过程?(回答'是'或'否')")

    flag1 = True
    while flag1:
        if interact1 == "是":
            print("==================================")
            answer = get_gcd(a, b, 1)
            break
        elif interact1 == "否":
            answer = get_gcd(a, b, 0)
            break
        else:
            print("您的输入有误，请正确回答!")
            interact1 = input("是否显示求最大公因数的过程?(回答'是'或'否')")

    print("==================================")
    print(f"{a} 与 {b} 的最大公约数为: {answer}")
    interact2 = input("是否显示辗转相除法的计算过程?(回答'是'或'否')")

    flag2 = True
    while flag2:
        if interact2 == "是":
            print("==================================")
            print(solve(remainder, answer, 1))
            break
        elif interact2 == "否":
            print(solve(remainder, answer))
            break
        else:
            print("您的输入有误，请正确回答!")
            interact2 = input("是否显示求最大公因数的过程?(回答'是'或'否')")


if __name__ == "__main__":
    equation, remainder, coefficient = [], [], []
    main()
