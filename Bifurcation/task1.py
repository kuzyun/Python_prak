import sys
import numpy as np
import sympy as sp
from sympy import diff

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
    a11 = diff(f1, x)
    a12 = diff(f1, y)
    a21 = diff(f2, x)
    a22 = diff(f2, y)
    spur = a11 + a22
    det = a11 * a22 - a12 * a21

    Y = sp.solve(f1, y)[0]
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
                det = det.subs(y, Y)
                prev1 = det.subs([(x, realxstep[len(realxstep) - 2]), (y, ystep[len(realxstep) - 2]), (k2, k2step[len(realxstep) - 2])])
                curr1 = det.subs([(x, realxstep[len(realxstep) - 1]), (y, ystep[len(realxstep) - 1]), (k2, k2step[len(realxstep) - 1])])
                # prev2 = spur.subs([(x, realxstep[len(realxstep) - 2]), (y, ystep[len(realxstep) - 2]), (k2, k2step[len(realxstep) - 2])])
                # curr2 = spur.subs([(x, realxstep[len(realxstep) - 1]), (y, ystep[len(realxstep) - 1]), (k2, k2step[len(realxstep) - 1])])
                if prev1 * curr1 < 0:# or prev2 * curr2 < 0:
                    dots.append(len(realxstep) - 2)

        else:
            break
    dots = dot_correct(det, dots, realxstep, ystep, k2step, K2)
    print(dots)
    param_plot(realxstep, ystep, k2step, subplot, dots)

def dot_correct(det, dots, xstep, ystep, kstep, K2):
    x, y, k2 = sp.symbols('x y k2')
    N = len(dots)
    res = []
    for i in dots:
        f1 = det.subs([(x, xstep[i]), (y, ystep[i]), (k2, kstep[i])])
        f2 = det.subs([(x, xstep[i + 1]), (y, ystep[i + 1]), (k2, kstep[i + 1])])
        bif = xstep[i] - f1 * (xstep[i + 1] - xstep[i]) / (f2 - f1)
        res.append((K2.subs(x, bif), bif))
    return res

def param_plot(xstep, ystep, k2step, subplot, dots):
    subplot.plot(k2step, ystep, 'r', k2step, xstep, 'g')
    for el in dots:
        subplot.plot(el[0], el[1], 'bx')
    subplot.set_ylabel('x, y')
    subplot.set_xlabel('k2')

def start_calc(N, subplot, canvas):
    subplot.cla()
    k3 = int(k3pick.get())
    alpha = int(alphapick.get())
    param_analysis(N, subplot, alpha=alpha, k3=k3)
    canvas.show()

root = Tk()

plot = Figure(figsize=(5, 4), dpi=100)
subplot = plot.add_subplot(111)
canvas = FigureCanvasTkAgg(plot, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=LEFT, expand=1)

combobox1_values = "\n".join(alphasdict.values())
alphapick = ttk.Combobox(root, values=combobox1_values)
alphapick.current(2)
alphapick.pack()

combobox2_values = "\n".join(k3dict.values())
k3pick = ttk.Combobox(root, values=combobox2_values)
k3pick.current(2)
k3pick.pack()

N = 100

button = Button(root, text='Запуск', command=lambda : start_calc(N, subplot, canvas)).pack(fill=X)

root.mainloop()