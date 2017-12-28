import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
cimport cython
from scipy import constants
from libc.math cimport pow


#Алгоритм Верле без typed memoryview без openmp
@cython.boundscheck(False)
@cython.wraparound(False)
def Verlet_Cython_nm_no(np.ndarray[double, ndim = 2] pos, np.ndarray[double, ndim = 2] vel, np.ndarray[double, ndim = 1] m, double time):
    print("Start Cython")
    cdef int N = len(m)
    cdef double G = constants.G
    cdef np.ndarray[double, ndim=1] t = np.linspace(0, time, 101)
    cdef double dt = t[1] - t[0]
    cdef np.ndarray[double, ndim=3] sol = np.zeros((len(t), N, 6))
    for i in range(N):
        sol[0][i][0] = pos[i][0]
        sol[0][i][1] = pos[i][1]
        sol[0][i][2] = pos[i][2]
        sol[0][i][3] = vel[i][0]
        sol[0][i][4] = vel[i][1]
        sol[0][i][5] = vel[i][2]
    # print("Sol ", 0, sol[0])
    cdef np.ndarray[double, ndim=2] A = np.zeros((N, 3))
    for j in range(N):
        ai1 = 0
        ai2 = 0
        ai3 = 0
        for k in range(N):
            if k != j:
                r = np.linalg.norm(
                    [sol[0][k][0] - sol[0][j][0], sol[0][k][1] - sol[0][j][1], sol[0][k][2] - sol[0][j][2]]) ** 3
                tmp = G * m[k] / r
                ai1 = ai1 + tmp * (sol[0][k][0] - sol[0][j][0])
                ai2 = ai2 + tmp * (sol[0][k][1] - sol[0][j][1])
                ai3 = ai3 + tmp * (sol[0][k][2] - sol[0][j][2])
        A[j] = [ai1, ai2, ai3]
    for i in range(1, len(t)):
        # print("A ", i, A)
        iter = []
        for j in range(N):
            iter.append(sol[i - 1][j][0] + sol[i - 1][j][3] * dt + 0.5 * A[j][0] * dt ** 2)
            iter.append(sol[i - 1][j][1] + sol[i - 1][j][4] * dt + 0.5 * A[j][1] * dt ** 2)
            iter.append(sol[i - 1][j][2] + sol[i - 1][j][5] * dt + 0.5 * A[j][2] * dt ** 2)
        F = []
        for j in range(N):
            F.append([])
            f1 = 0
            f2 = 0
            f3 = 0
            for k in range(N):
                if k != j:
                    r = np.linalg.norm([iter[k * 3] - iter[j * 3], iter[k * 3 + 1] - iter[j * 3 + 1], iter[k * 3 + 2] - iter[j * 3 + 2]]) ** 3
                    tmp = G * m[k] / r
                    f1 = f1 + tmp * (iter[k * 3] - iter[j * 3])
                    f2 = f2 + tmp * (iter[k * 3 + 1] - iter[j * 3 + 1])
                    f3 = f3 + tmp * (iter[k * 3 + 2] - iter[j * 3 + 2])
            F[j].extend([f1, f2, f3])
            sol[i][j] = [iter[j * 3], iter[j * 3 + 1], iter[j * 3 + 2], sol[i - 1][j][3] + 0.5 * (f1 + A[j][0]) * dt,
                       sol[i - 1][j][4] + 0.5 * (f2 + A[j][1]) * dt, sol[i - 1][j][5] + 0.5 * (f3 + A[j][2]) * dt]
        A = np.array(F)
    return sol

#Алгоритм Верле с typed memoryview без openmp
@cython.boundscheck(False)
@cython.wraparound(False)
def Verlet_Cython_m_no(double[:, :] pos, double[:, :] vel, double[:] m, double time):
    print("Start Cython")
    cdef int N = len(m)
    cdef double G = constants.G
    cdef double[:] t = np.linspace(0, time, 101)
    cdef double dt = t[1] - t[0]
    cdef double[:, :, :] sol = np.zeros((len(t), N, 6))
    for i in range(N):
        sol[0, i, 0] = pos[i, 0]
        sol[0, i, 1] = pos[i, 1]
        sol[0, i, 2] = pos[i, 2]
        sol[0, i, 3] = vel[i, 0]
        sol[0, i, 4] = vel[i, 1]
        sol[0, i, 5] = vel[i, 2]
    # print("Sol ", 0, sol[0])
    cdef double[:, :] A = np.zeros((N, 3))
    for j in range(N):
        ai1 = 0
        ai2 = 0
        ai3 = 0
        for k in range(N):
            if k != j:
                r = np.linalg.norm(
                    [sol[0, k, 0] - sol[0, j, 0], sol[0, k, 1] - sol[0, j, 1], sol[0, k, 2] - sol[0, j, 2]]) ** 3
                tmp = G * m[k] / r
                ai1 += tmp * (sol[0, k, 0] - sol[0, j, 0])
                ai2 += tmp * (sol[0, k, 1] - sol[0, j, 1])
                ai3 += tmp * (sol[0, k, 2] - sol[0, j, 2])
        A[j][0] = ai1
        A[j][1] = ai2
        A[j][2] = ai3
    cdef double[:, :] iter  = np.zeros((N, 3))
    for i in range(1, len(t)):
        for j in range(N):
            iter[j, 0] = (sol[i - 1, j, 0] + sol[i - 1, j, 3] * dt + 0.5 * A[j, 0] * dt ** 2)
            iter[j, 1] = (sol[i - 1, j, 1] + sol[i - 1, j, 4] * dt + 0.5 * A[j, 1] * dt ** 2)
            iter[j, 2] = (sol[i - 1, j, 2] + sol[i - 1, j, 5] * dt + 0.5 * A[j, 2] * dt ** 2)
        for j in range(N):
            f1 = 0
            f2 = 0
            f3 = 0
            for k in range(N):
                if k != j:
                    r = np.linalg.norm([iter[k, 0] - iter[j, 0], iter[k, 1] - iter[j, 1], iter[k, 2] - iter[j, 2]]) ** 3
                    tmp = G * m[k] / r
                    f1 += tmp * (iter[k, 0] - iter[j, 0])
                    f2 += tmp * (iter[k, 1] - iter[j, 1])
                    f3 += tmp * (iter[k, 2] - iter[j, 2])
            sol[i, j, 0] = iter[j, 0]
            sol[i, j, 1] = iter[j, 1]
            sol[i, j, 2] = iter[j, 2]
            sol[i, j, 3] = sol[(i - 1), j, 3] + 0.5 * (f1 + A[j, 0]) * dt
            sol[i, j, 4] = sol[(i - 1), j, 4] + 0.5 * (f2 + A[j, 1]) * dt
            sol[i, j, 5] = sol[(i - 1), j, 5] + 0.5 * (f3 + A[j, 2]) * dt
            A[j][0] = f1
            A[j][1] = f2
            A[j][2] = f3
    return sol

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double norm(double x, double y, double z) nogil:
    cdef double r = 0
    r = pow(x * x + y * y + z * z, 0.5)
    return r

#Алгоритм Верле с typed memoryview с openmp
@cython.boundscheck(False)
@cython.wraparound(False)
def Verlet_Cython_m_o(double[:, :] pos, double[:, :] vel, double[:] m, double time):
    print("Start Cython")
    cdef:
        int N = len(m)
        double G = constants.G
        double[:] t = np.linspace(0, time, 101)
        double dt = t[1] - t[0]
        double[:, :, :] sol = np.zeros((len(t), N, 6))
        double[:, :] A = np.zeros((N, 3))
        double[:, :] iter  = np.zeros((N, 3))
        int i, j, k
        double f1, f2, f3
        double ai1, ai2, ai3
        double tmp, r

    for i in prange(N, nogil=True):
        sol[0, i, 0] = pos[i, 0]
        sol[0, i, 1] = pos[i, 1]
        sol[0, i, 2] = pos[i, 2]
        sol[0, i, 3] = vel[i, 0]
        sol[0, i, 4] = vel[i, 1]
        sol[0, i, 5] = vel[i, 2]
    # print("Sol ", 0, sol[0])
    for j in range(N):
        ai1 = 0
        ai2 = 0
        ai3 = 0
        for k in prange(N, nogil=True):
            if k != j:
                r = norm(sol[0, k, 0] - sol[0, j, 0], sol[0, k, 1] - sol[0, j, 1], sol[0, k, 2] - sol[0, j, 2]) ** 3
                tmp = G * m[k] / r
                ai1 += tmp * (sol[0, k, 0] - sol[0, j, 0])
                ai2 += tmp * (sol[0, k, 1] - sol[0, j, 1])
                ai3 += tmp * (sol[0, k, 2] - sol[0, j, 2])
        A[j][0] = ai1
        A[j][1] = ai2
        A[j][2] = ai3
    for i in range(1, len(t)):
        for j in prange(N, nogil=True):
            iter[j, 0] = (sol[i - 1, j, 0] + sol[i - 1, j, 3] * dt + 0.5 * A[j, 0] * dt ** 2)
            iter[j, 1] = (sol[i - 1, j, 1] + sol[i - 1, j, 4] * dt + 0.5 * A[j, 1] * dt ** 2)
            iter[j, 2] = (sol[i - 1, j, 2] + sol[i - 1, j, 5] * dt + 0.5 * A[j, 2] * dt ** 2)
        for j in range(N):
            f1 = 0
            f2 = 0
            f3 = 0
            for k in prange(N, nogil=True):
                if k != j:
                    r = norm(iter[k, 0] - iter[j, 0], iter[k, 1] - iter[j, 1], iter[k, 2] - iter[j, 2]) ** 3
                    tmp = G * m[k] / r
                    f1 += tmp * (iter[k, 0] - iter[j, 0])
                    f2 += tmp * (iter[k, 1] - iter[j, 1])
                    f3 += tmp * (iter[k, 2] - iter[j, 2])
            sol[i, j, 0] = iter[j, 0]
            sol[i, j, 1] = iter[j, 1]
            sol[i, j, 2] = iter[j, 2]
            sol[i, j, 3] = sol[(i - 1), j, 3] + 0.5 * (f1 + A[j, 0]) * dt
            sol[i, j, 4] = sol[(i - 1), j, 4] + 0.5 * (f2 + A[j, 1]) * dt
            sol[i, j, 5] = sol[(i - 1), j, 5] + 0.5 * (f3 + A[j, 2]) * dt
            A[j][0] = f1
            A[j][1] = f2
            A[j][2] = f3
    return sol

#Алгоритм Верле без typed memoryview c openmp
@cython.boundscheck(False)
@cython.wraparound(False)
def Verlet_Cython_nm_o(np.ndarray[double, ndim = 2] pos, np.ndarray[double, ndim = 2] vel, np.ndarray[double, ndim = 1] m, time):
    print("Start Cython")
    cdef :
        int N = len(m)
        double G = constants.G
        np.ndarray[double, ndim=1] t = np.linspace(0, time, 101)
        double dt = t[1] - t[0]
        np.ndarray[double, ndim=3] sol = np.zeros((len(t), N, 6))
        np.ndarray[double, ndim=2] A = np.zeros((N, 3))
        np.ndarray[double, ndim=2] iter = np.zeros((N, 3))
        int i, j, k
        double tmp, r
        double f1, f2, f3
        double ai1, ai2, ai3

    for i in prange(N, nogil=True):
        sol[0, i, 0] = pos[i, 0]
        sol[0, i, 1] = pos[i, 1]
        sol[0, i, 2] = pos[i, 2]
        sol[0, i, 3] = vel[i, 0]
        sol[0, i, 4] = vel[i, 1]
        sol[0, i, 5] = vel[i, 2]
    for j in range(N):
        ai1 = 0
        ai2 = 0
        ai3 = 0
        for k in prange(N, nogil=True):
            if k != j:
                r = norm(sol[0, k, 0] - sol[0, j, 0], sol[0, k, 1] - sol[0, j, 1], sol[0, k, 2] - sol[0, j, 2]) ** 3
                tmp = G * m[k] / r
                ai1 += tmp * (sol[0, k, 0] - sol[0, j, 0])
                ai2 += tmp * (sol[0, k, 1] - sol[0, j, 1])
                ai3 += tmp * (sol[0, k, 2] - sol[0, j, 2])
        A[j][0] = ai1
        A[j][1] = ai2
        A[j][2] = ai3
    for i in range(1, len(t)):
        for j in prange(N, nogil=True):
            iter[j, 0] = (sol[i - 1, j, 0] + sol[i - 1, j, 3] * dt + 0.5 * A[j, 0] * dt ** 2)
            iter[j, 1] = (sol[i - 1, j, 1] + sol[i - 1, j, 4] * dt + 0.5 * A[j, 1] * dt ** 2)
            iter[j, 2] = (sol[i - 1, j, 2] + sol[i - 1, j, 5] * dt + 0.5 * A[j, 2] * dt ** 2)
        for j in range(N):
            f1 = 0
            f2 = 0
            f3 = 0
            for k in prange(N, nogil=True):
                if k != j:
                    r = norm(iter[k, 0] - iter[j, 0], iter[k, 1] - iter[j, 1], iter[k, 2] - iter[j, 2]) ** 3
                    tmp = G * m[k] / r
                    f1 += tmp * (iter[k, 0] - iter[j, 0])
                    f2 += tmp * (iter[k, 1] - iter[j, 1])
                    f3 += tmp * (iter[k, 2] - iter[j, 2])
            sol[i, j, 0] = iter[j, 0]
            sol[i, j, 1] = iter[j, 1]
            sol[i, j, 2] = iter[j, 2]
            sol[i, j, 3] = sol[(i - 1), j, 3] + 0.5 * (f1 + A[j, 0]) * dt
            sol[i, j, 4] = sol[(i - 1), j, 4] + 0.5 * (f2 + A[j, 1]) * dt
            sol[i, j, 5] = sol[(i - 1), j, 5] + 0.5 * (f3 + A[j, 2]) * dt
            A[j][0] = f1
            A[j][1] = f2
            A[j][2] = f3
    return sol