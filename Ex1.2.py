import sys
import numpy
import cmath

#пересекаются ли отрезки
#[a1, a2] - первый отрезок
#[b1, b2] - второй отрезок
def isinrange(a, b):
    if min(a[0], a[1]) > max(b[0], b[1]) or max(a[0], a[1]) < min(b[0], b[1]):
        return "Не пересекаются"
    return [max(min(a[0], a[1]), min(b[0], b[1])), min(max(a[0], a[1]), max(b[0], b[1]))]

#Пересекаются
# Tr1 = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
# Tr2 = [[0, 0, 0], [3, 0, 0], [3, 2, 0]]

Tr1 = [[0, 0, 0], [3, 0, 0], [0, 3, 0]]
Tr2 = [[5, 1, 0],[1, 1, 1],[2, 3, 5]]

#Не пересекаются
# Tr1 = [[0, 0, 0], [1, 1, 2], [2, 0, 0]]
# Tr2 = [[1, 1, 0], [3, 2, 0], [1, 3, 0]]
i = 0
N1 = 0
N2 = 0

def Tr1Plain(X):
    d1 = - numpy.dot(N1, Tr1[0])
    return numpy.dot(N1, X) + d1

def Tr2Plain(X):
    d2 = - numpy.dot(N2, Tr2[0])
    return numpy.dot(N2, X) + d2

def VecAbs(vector):
    return cmath.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).real

N1 = numpy.cross([Tr1[1][0] - Tr1[0][0], Tr1[1][1] - Tr1[0][1], Tr1[1][2] - Tr1[0][2]],
                 [Tr1[2][0] - Tr1[0][0], Tr1[2][1] - Tr1[0][1], Tr1[2][2] - Tr1[0][2]])

N2 = numpy.cross([Tr2[1][0] - Tr2[0][0], Tr2[1][1] - Tr2[0][1], Tr2[1][2] - Tr2[0][2]],
                 [Tr2[2][0] - Tr2[0][0], Tr2[2][1] - Tr2[0][1], Tr2[2][2] - Tr2[0][2]])
print()
d1 = [0, 0, 0]
d2 = [0, 0, 0]
for i in range(0, 3):
    d1[i] = Tr2Plain(Tr1[i])
    d2[i] = Tr1Plain(Tr2[i])
if(d1[0] * d1[1] > 0 and d1[1] * d1[2] > 0 and d1[0] * d1[2] > 0 and d1[0] != 0 and d1[1] != 0 and d1[2] != 0):
    print("Треугольники не пересекаются")
    sys.exit(0)
if (d2[0] * d2[1] > 0 and d2[1] * d2[2] > 0 and d2[0] * d2[2] > 0 and d2[0] != 0 and d2[1] != 0 and d2[2] != 0):
    print("Треугольники не пересекаются")
    sys.exit(0)
if (d1[0] == 0 or d1[1] == 0 or d1[2] == 0):
    for i in range(0, 3):
        if d1[i] == 0:
            bar = []
            for j in range(0, 3):
                if [Tr2[j][0] - Tr1[i][0], Tr2[j][1] - Tr1[i][1], Tr2[j][2] - Tr1[i][2]] == [0, 0, 0] or [Tr2[(j + 1) % 3][0] - Tr1[i][0], Tr2[(j + 1) % 3][1] - Tr1[i][1], Tr2[(j + 1) % 3][2] - Tr1[i][2]] == [0, 0, 0]:
                    print("Треугольники пересекаются")
                    sys.exit(0)
                bar.append(float(numpy.dot([Tr2[j][0] - Tr1[i][0], Tr2[j][1] - Tr1[i][1], Tr2[j][2] - Tr1[i][2]],
                            [Tr2[(j + 1) % 3][0] - Tr1[i][0], Tr2[(j + 1) % 3][1] - Tr1[i][1], Tr2[(j + 1) % 3][2] - Tr1[i][2]])
                           / VecAbs([Tr2[j][0] - Tr1[i][0], Tr2[j][1] - Tr1[i][1], Tr2[j][2] - Tr1[i][2]])
                           / VecAbs([Tr2[(j + 1) % 3][0] - Tr1[i][0], Tr2[(j + 1) % 3][1] - Tr1[i][1], Tr2[(j + 1) % 3][2] - Tr1[i][2]])))
                k = 0
                for l in bar:
                    if l < 0:
                        k = k + 1
                if k >= 2:
                    print("Треугольники пересекаются")
                    sys.exit(0)
    print("Треугольники не пересекаются")
    sys.exit(0)
if (d2[0] == 0 or d2[1] == 0 or d2[2] == 0):
    for i in range(0, 3):
        if d2[i] == 0:
            bar = []
            for j in range(0, 3):
                # if [Tr1[j][0] - Tr2[i][0], Tr1[j][1] - Tr2[i][1], Tr1[j][2] - Tr2[i][2]] == [0, 0, 0] or [Tr1[(j + 1) % 3][0] - Tr1[i][0], Tr1[(j + 1) % 3][1] - Tr1[i][1], Tr1[(j + 1) % 3][2] - Tr1[i][2]] == [0, 0, 0]:
                #     print("Треугольники пересекаются")
                #     sys.exit(0)
                bar.append(float(numpy.dot([Tr1[j][0] - Tr2[i][0], Tr1[j][1] - Tr2[i][1], Tr1[j][2] - Tr2[i][2]],
                                           [Tr1[(j + 1) % 3][0] - Tr2[i][0], Tr1[(j + 1) % 3][1] - Tr2[i][1],
                                            Tr1[(j + 1) % 3][2] - Tr2[i][2]])
                                 / VecAbs([Tr1[j][0] - Tr2[i][0], Tr1[j][1] - Tr2[i][1], Tr1[j][2] - Tr2[i][2]])
                                 / VecAbs([Tr1[(j + 1) % 3][0] - Tr2[i][0], Tr1[(j + 1) % 3][1] - Tr2[i][1], Tr1[(j + 1) % 3][2] - Tr2[i][2]])))
                k = 0
                print(bar)
                for l in bar:
                    if l < 0:
                        k = k + 1
                if k >= 2:
                    print("Треугольники пересекаются")
                    sys.exit(0)
    print("Треугольники не пересекаются")
    sys.exit(0)
ind1 = 0
ind2 = 0
if d1[0] * d1[1] >= 0:
    ind1 = 2
if d1[0] * d1[2] > 0:
    ind1 = 1
if d1[2] * d1[1] > 0:
    ind1 = 0

if d2[0] * d2[1] > 0:
    ind2 = 2
if d2[0] * d2[2] > 0:
    ind2 = 1
if d2[2] * d2[1] > 0:
    ind2 = 0
D = numpy.cross(N1, N2)
p1 = [0, 0, 0]
p2 = [0, 0, 0]
for i in range(0, 3):
    if max(abs(D[0]), abs(D[1]), abs(D[2])) == D[0]:
        p1[i] = Tr1[i][0]
        p2[i] = Tr2[i][0]
    if max(abs(D[0]), abs(D[1]), abs(D[2])) == D[1]:
        p1[i] = Tr1[i][1]
        p2[i] = Tr2[i][1]
    if max(abs(D[0]), abs(D[1]), abs(D[2])) == D[2]:
        p1[i] = Tr1[i][2]
        p2[i] = Tr2[i][2]
range1 = []
range2 = []
for i in range(0, 3):
    if i != ind1:
        t1 = p1[i] + (p1[ind1] - p1[i]) * float(d1[i] / (d1[i] - d1[ind1]))
        range1.append(t1)
    if i != ind2:
        t2 = p2[i] + (p2[ind2] - p2[i]) * float(d2[i] / (d2[i] - d2[ind2]))
        range2.append(t2)
if isinstance(isinrange(range1, range2), str):
    print("Треугольники не пересекаются")
    sys.exit(0)
print("Треугольники пересекаются")