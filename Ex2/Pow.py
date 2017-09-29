def pow(c, n):
    if n == 0:
        return 1
    if n % 2:
        return c * pow(c, n - 1)
    return pow(c * c, n / 2)

c = int(input("Введите число:\n"))
n = int(input("Введите степень числа:\n"))
print(pow(c, n))