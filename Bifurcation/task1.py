import sys
import numpy as np
import sympy as sp
from sympy import diff
from scipy import integrate

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.axes import Axes

import tkinter.ttk as ttk
from tkinter import *
import cmath

#Параметры по умолчанию
_alpha = 18
_k1 = 0.012
_k_1 = 0.01
_k2 = 0.012
_k_2 = 10e-9
_k3 = 10

#словарь альф
alphasdict = {0: '10', 1: '15', 2: '18', 3: '20', 4:'25'}

#словарь k3
k3dict = {0: '1', 1: '5', 2: '10', 3: '50', 4:'100'}

def param_analysis(N, subplot, alpha=_alpha, k1=_k1, k_1=_k_1, k2=_k2, k_2=_k_2, k3=_k3):
    xstep = np.linspace(0, 1, N)
    x, y, k2 = sp.symbols('x y k2')
    #матрица Якоби
    f1 = k1 * (1 - x - y) - k_1 * x - k3 * (1 - x)**alpha * x * y
    f2 = k2 * (1 - x - y) ** 2 - k_2 * y**2 - k3 * (1 - x)**alpha * x * y

    # след и определитель Якобиана
    A = sp.Matrix([f1, f2])
    jacA = A.jacobian([x, y])
    det_jacA = jacA.det()
    trace = sp.trace(jacA)

    Y = sp.solve(f1, y)[0]
    print(Y)
    # Y = (k1 - k1 * x - k_1 * x) / (k1 + k3 * (1 - x)**2 * x)
    # K2 = (k_2 * Y**2 + k3 * (1-x)**alpha*x*Y) / (1-x-Y)**2
    K2 = sp.solve(f2, k2)[0]
    K2 = K2.subs(y, Y)
    ystep=[]
    k2step=[]
    realxstep = []
    dots = []
    for i in xstep:
        tmpy = Y.subs(x, i)
        if(tmpy >= 0 and tmpy <= 1 and tmpy + i >= 0 and tmpy + i <= 1):
            if K2.subs(x, i) != sp.zoo:
                k2step.append(K2.subs(x, i))
                ystep.append(tmpy)
                realxstep.append(i)
            if i > 0:
                det = det_jacA.subs(y, Y)
                prev = det.subs([(x, realxstep[len(realxstep) - 2]), (y, ystep[len(realxstep) - 2]), (k2, k2step[len(realxstep) - 2])])
                curr = det.subs([(x, realxstep[len(realxstep) - 1]), (y, ystep[len(realxstep) - 1]), (k2, k2step[len(realxstep) - 1])])
                if prev * curr < 0:
                    dots.extend(dot_correct(det, len(realxstep) - 2, realxstep, ystep, k2step, K2, Y))

            if i > 0:
                det = det_jacA.subs(y, Y)
                prev = trace.subs([(x, realxstep[len(realxstep) - 2]), (y, ystep[len(realxstep) - 2]), (k2, k2step[len(realxstep) - 2])])
                curr = trace.subs([(x, realxstep[len(realxstep) - 1]), (y, ystep[len(realxstep) - 1]), (k2, k2step[len(realxstep) - 1])])
                if prev * curr < 0:
                    dots.extend(dot_correct(trace, len(realxstep) - 2, realxstep, ystep, k2step, K2, Y))
        else:
            break
    # dots = dot_correct(det, dots, realxstep, ystep, k2step, K2)
    print(dots)
    # param_plot(realxstep, ystep, k2step, subplot, dots)
    param_plot(realxstep, k2step, subplot, 'r', 0, 0, "$x_{k_2}$")
    param_plot(ystep, k2step, subplot, 'g', 0, 0, "$y_{k_2}$")
    dot_plot(subplot, dots)

def duoparam_analysis(N, subplot, alpha=_alpha, k1=_k1, k_1=_k_1, k2=_k2, k_2=_k_2, k3=_k3):
    xstep = np.linspace(0, 1, N)
    x, y, k2, k1 = sp.symbols('x y k2 k1')
    f1 = k1 * (1 - x - y) - k_1 * x - k3 * (1 - x)**alpha * x * y
    f2 = k2 * (1 - x - y) ** 2 - k_2 * y**2 - k3 * (1 - x)**alpha * x * y

    eq1 = sp.lambdify((x, y, k1), f1)
    eq2 = sp.lambdify((x, y, k2), f2)
    Y, X = np.mgrid[0:.5:1000j, 0:1:2000j]

    U = eq1(X, Y, 0.012)
    V = eq2(X, Y, 0.012)
    print("HERE")
    velocity = np.sqrt(U * U + V * V)
    plt.streamplot(X, Y, U, V, density=[2.5, 0.8], color=velocity)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xlim([0.1, 0.5])
    # plt.ylim([0.1, 0.5])
    # plt.title('Phase portrait')
    plt.show()

    #след и определитель Якобиана
    A = sp.Matrix([f1, f2])
    jacA = A.jacobian([x, y])
    det_jacA = jacA.det()
    trace = sp.trace(jacA)

    Y = sp.solve(f1, y)[0]
    Det = sp.solve(det_jacA.subs(y, Y), k2)[0]
    Trace = sp.solve(trace.subs(y, Y), k2)[0]
    K2 = sp.solve(f2.subs(y, Y), k2)[0]
    res = Det - K2
    tr = sp.solve(Trace - K2, k1)[0]
    print("3.5: ", res)
    print("tr", tr)
    # print("3.5: ", sp.expand(Det - K2))
    # K1 = sp.solve((Det - K2), k1)[0]
    # print("4 ", K1)

    k1step_det = []
    k2step_det = []
    k1step_tr = []
    k2step_tr = []

    dots = []
    for i in xstep:
        sol1 = sp.solve(res.subs(x, i), k1)
        # print("sol", sol1)
        if len(sol1) > 0:
            tmpk1_det = sol1[0]
            tmpk2_det = Det.subs([(k1, tmpk1_det), (x, i)])
            # print("tmpk1", tmpk1_det)
            # print("tmpk2", tmpk2_det)
            k1step_det.append(sp.re(tmpk1_det))
            k2step_det.append(sp.re(tmpk2_det))

        tmpk1_tr = tr.subs(x, i)
        tmpk2_tr = Trace.subs([(k1, tmpk1_tr), (x, i)])
        k1step_tr.append(sp.re(tmpk1_tr))
        k2step_tr.append(sp.re(tmpk2_tr))
        print(i)
    print("k1det", k1step_det)
    print("k1tr", k1step_tr)
    param_plot(k1step_det, k2step_det, subplot1, 'g', [0, 1], 0)
    param_plot(k1step_tr, k2step_tr, subplot1, 'b', [0, 1], 0)
    #(0.06, 0.02)

def dot_correct(det, i, xstep, ystep, kstep, K2, Y):
    x, y, k2 = sp.symbols('x y k2')
    res = []
    # for i in dots:
    f1 = det.subs([(x, xstep[i]), (y, ystep[i]), (k2, kstep[i])])
    f2 = det.subs([(x, xstep[i + 1]), (y, ystep[i + 1]), (k2, kstep[i + 1])])
    bif = xstep[i] - f1 * (xstep[i + 1] - xstep[i]) / (f2 - f1)
    # bif = ystep[i] - f1 * (ystep[i + 1] - ystep[i]) / (f2 - f1)
    res.append((K2.subs(x, bif), bif, 'rx'))
    res.append((K2.subs(x, bif), Y.subs(x, bif), 'gx'))
    return res

def param_plot(xstep, k2step, subplot, color, limx, limy, label):
    subplot.plot(k2step, xstep, color, label=label)
    subplot.legend()
    subplot.set_ylabel('$x, y$')
    subplot.set_xlabel('$k_2$')
    subplot.set_xlim(limx)
    subplot.set_ylim(limy)

def dot_plot(subplot, dots):
    for el in dots:
        subplot.plot(el[0], el[1], el[2])

def func(init, t, alpha=_alpha, k1=_k1, k_1=_k_1, k2=_k2, k_2=_k_2, k3=_k3):
    x0 = init[0]
    y0 = init[1]
    k1 = 0.012
    k2 = 0.012
    f1 = k1 * (1 - x0 - y0) - k_1 * x0 - k3 * (1 - x0)**alpha * x0 * y0
    f2 = k2 * (1 - x0 - y0) ** 2 - k_2 * y0**2 - k3 * (1 - x0)**alpha * x0 * y0
    return [f1, f2]

def oscillations():
    t = np.linspace(0, 5000, 1000)
    init = [0.1, 0.1]
    sol = integrate.odeint(func, init, t)
    plt.plot(t, sol[:, 0], color='r', label="$x(t)$")
    plt.plot(t, sol[:, 1], color='g', label="$y(t)$")
    plt.legend()
    plt.show()

def start_calc(N, subplot, canvas, subplot1, canvas1):
    subplot.cla()
    subplot1.cla()
    k3 = int(k3pick.get())
    alpha = int(alphapick.get())
    param_analysis(N, subplot, alpha=alpha, k3=k3)
    duoparam_analysis(N, subplot1, alpha=alpha, k3=k3)
    oscillations()
    canvas.show()
    canvas1.show()

root = Tk()

plot = Figure(figsize=(5, 4), dpi=100)
subplot = plot.add_subplot(111)
canvas = FigureCanvasTkAgg(plot, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=LEFT, expand=1)

plot1 = Figure(figsize=(5, 4), dpi=100)
subplot1 = plot1.add_subplot(111)
canvas1 = FigureCanvasTkAgg(plot1, master=root)
canvas1.show()
canvas1.get_tk_widget().pack(side=LEFT, expand=1)

combobox1_values = "\n".join(alphasdict.values())
alphapick = ttk.Combobox(root, values=combobox1_values)
alphapick.current(2)
alphapick.pack()

combobox2_values = "\n".join(k3dict.values())
k3pick = ttk.Combobox(root, values=combobox2_values)
k3pick.current(2)
k3pick.pack()

N = 100

button1 = Button(root, text='Запуск', command=lambda : start_calc(N, subplot, canvas, subplot1, canvas1)).pack(fill=X)

root.mainloop()