import sys
import numpy as np
from numpy import random
import multiprocessing as mp
import threading as thr
import time

def mult(A, B):
    n = len(A)
    m = len(A[0])
    k = len(B[0])
    res = []
    for i in range(n):
        res.append([])
        for j in range(k):
            sum = 0
            for l in range(m):
                sum = sum + A[i][l] * B[l][j]
            res[i].append(sum)
    return res

def mult1(A, B, i, j, q):
    # n = len(A)
    m = len(A[0])
    k = len(B[0])
    res = []
    # for i in range(n):
    # res.append([])
    # for j in range(k):
    sum = 0
    for l in range(m):
        sum = sum + A[i][l] * B[l][j]
    res.append(sum)
    q.put(res)

n, m, k = 3, 4, 5
A = random.random((n, m))
B = random.random((m, k))
res = []
for i in range(n):
    res.append([])
    for j in range(k):
        res[i].append(0)
# res = mult(A, B)
# print(res)
if __name__ == '__main__':
    for i in range(n):
        for j in range(k):
            q = mp.Queue()
            p = mp.Process(target=mult1, args=(A, B, i, j, q))
            p.start()
            res[i][j] = q.get()
            p.join()

    print(res)
    # res1 = np.dot(A, B)
    res = mult(A, B)
    print(res)
    # print(res)