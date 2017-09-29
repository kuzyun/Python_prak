import sys
import numpy
def incseq(L):
    young = []
    try:
        if not isinstance(L[0], int):
            raise TypeError
    except TypeError:
        print("Элемент не является целым числом")
        sys.exit(1)
    young.append([L[0]])
    for i in range(1, len(L)):
        try:
            if not isinstance(L[i], int):
                raise TypeError
        except TypeError:
            print("Элемент не является целым числом")
            sys.exit(1)
        j = 0
        k = L[i]
        while 1:
            if j == len(young):
                young.append([k])
                break
            if k > young[j][len(young[j]) - 1]:
                young[j].append(k)
                break
            else:
                num = numpy.searchsorted(young[j], k)
                tmp = young[j][num]
                young[j][num] = k
                k = tmp
                j = j + 1
    return len(young[0])

L = [2, 1, 3, 2, 4, 3]
print(incseq(L))