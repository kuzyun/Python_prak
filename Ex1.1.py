import sys
import random
N = 100
k = 90
L = []
M = []
i = 0
while i < k:
    L.append(random.randint(0, N))
    i = i + 1
i = 0
while i < N + 1:
    M.append(0)
    i = i + 1
for l in L:
    M[l] = 1
i = 0
for l in M:
    i += l
print(i)
