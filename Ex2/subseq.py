import sys

def lcs(a, b):
    table = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                table[i + 1][j + 1] = table[i][j] + 1
            else:
                table[i + 1][j + 1] = max(table[i + 1][j], table[i][j + 1])
    return table[len(a)][len(b)]

a = input("Введите первую последовательность:\n")
b = input("Введите вторую последовательность:\n")
print("Длина наибольшей общей подпоследовательности:" + str(lcs(a, b)))