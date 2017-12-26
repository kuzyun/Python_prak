import sys
import ctypes
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy import constants
from scipy import integrate
import time
import threading
from queue import Queue
import multiprocessing
from multiprocessing import Process, sharedctypes
import mpl_toolkits.mplot3d as a3
import random

colorsdict = {0: 'Red', 1: 'Blue', 2: 'Green', 3: 'Grey', 4: 'Black'}
class CircleGUI:
    circles = 0
    def __init__(self, master):
        self.master = master
        self.master.title("Task 3")
        self.circles = CircleList
        plot = PlotArea(self.master)
        buttons = Buttons(master, plot)
        t = StringVar()
        lb = Label(master, textvariable=t).pack()
        t.set("Computation time:")
        tabs = Tabs(master)
        labels = AxesLabels(tabs.Edit)
        plot.canvas.mpl_connect('motion_notify_event', lambda event: OnMove(event, coord=labels))
        plot.canvas.mpl_connect('axes_leave_event', lambda event: OnLeave(event, coord=labels))
        colorpick = Colorpick(tabs.Edit)
        slider = Slider(tabs.Edit)
        plot.canvas.mpl_connect('button_press_event', lambda event: OnClick(event))
        openbutton = Button(master, text="Open", command=lambda : self.OpenFile(plot=plot, _slider=slider, _colorpick=colorpick))
        openbutton.pack(fill=X)
        savebutton = Button(master, text="Save", command=lambda : self.SaveFile(circles=self.circles, plot=plot,
                                                                                colorpick=colorpick, slider=slider))
        savebutton.pack(fill=X)
        rb = RadioButt(tabs.Model, plot, t)

        def OnMove(event, coord):
            x, y = event.xdata, event.ydata
            coord.xcoord.config(text=x)
            coord.ycoord.config(text=y)

        def OnLeave(event, coord):
            coord.xcoord.config(text="")
            coord.ycoord.config(text="")

        def OnClick(event):
            # plot.subplot.cla()
            Axes.set_xlim(plot.subplot, -200, 200)
            Axes.set_ylim(plot.subplot, -200, 200)
            x, y = labels.xcoord.cget("text"), labels.ycoord.cget("text")
            radius = slider.slider.get()
            colour = colorpick.colorpick.get()

            self.circles.addcircle(self.circles, radius, x, y, colour)
            circle = pyplot.Circle((x, y), radius=radius, color=colour)
            plot.subplot.add_patch(circle)
            plot.canvas.show()

    def OpenFile(event, plot, _slider, _colorpick):
        plot.subplot.cla()
        filename = askopenfilename(filetypes=[("XML files", "*.xml")])
        tree = ET.parse(filename)
        CircleGUI.circles = []
        for node in tree.iter('settings'):
            xlim = float(node.attrib.get('xlim'))
            ylim = float(node.attrib.get('ylim'))
            color = node.attrib.get('color')
            slider = float(node.attrib.get('slider'))
            Axes.set_xlim(plot.subplot, -xlim, xlim)
            Axes.set_ylim(plot.subplot, -ylim, ylim)
            _colorpick.colorpick.current(colorsdict.get(color))
            _slider.slider.set(slider)

        for node in tree.iter('circle'):
            x = float(node.attrib.get('x'))
            y = float(node.attrib.get('y'))
            r = float(node.attrib.get('radius'))
            color = node.attrib.get('color')
            CircleGUI.circles.append({'r': r, 'x': x, 'y': y, 'color': color})
            circle = pyplot.Circle((x, y), radius=r, color=color)
            plot.subplot.add_patch(circle)
        plot.canvas.show()

    def SaveFile(event, circles, plot, colorpick, slider):
        root = ET.Element('data')
        ET.SubElement(root, 'settings', {'xlim': str(Axes.get_xbound(plot.subplot)[1]), 'ylim': str(Axes.get_ybound(plot.subplot)[1]),
                                         'color': colorpick.colorpick.get(), 'slider': str(slider.slider.get())})
        # print(type(Axes.get_xbound(plot.subplot)))
        for el in circles.circlelist:
            child = ET.SubElement(root, 'circle', {'x': str(el['x']), 'y': str(el['y']), 'radius': str(el['r']), 'color': el['color']})
        # ET.dump(root)
        filename = asksaveasfilename(filetypes=[("XML files", "*.xml")])
        tree = ET.ElementTree(root)
        tree.write(filename)


class PlotArea:
    canvas = 0
    subplot = 0
    def __init__(self, master):
        plot = Figure(figsize=(5, 4), dpi=100)
        self.subplot = plot.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(plot, master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=LEFT, expand=1)
        Axes.set_xlim(self.subplot, -200, 200)
        Axes.set_ylim(self.subplot, -200, 200)

class Buttons:
    plusbutton = 0
    minusbutton = 0
    def __init__(self, master, plot):
        plusbutton = Button(master, text="-", command=lambda : self.PlusAct(plot=plot))
        minusbutton = Button(master, text="+", command=lambda: self.MunusAct(plot=plot))
        plusbutton.pack(fill=X)
        minusbutton.pack(fill=X)

    def PlusAct(event, plot):
        range = Axes.get_xbound(plot.subplot)
        range = tuple(1.5 * x for x in range)
        Axes.set_xbound(plot.subplot, range)
        Axes.set_ybound(plot.subplot, range)
        plot.canvas.show()

    def MunusAct(event, plot):
        range = Axes.get_xbound(plot.subplot)
        range = tuple(x / 1.5 for x in range)
        Axes.set_xbound(plot.subplot, range)
        Axes.set_ybound(plot.subplot, range)
        plot.canvas.show()

class AxesLabels:
    xcoord = 0
    ycoord = 0
    def __init__(self, master):
        self.xcoord = Label(master)
        self.ycoord = Label(master)
        self.xcoord.pack()
        self.ycoord.pack()

class Tabs:
    Edit = 0
    Model = 0
    def __init__(self, master):
        nb = ttk.Notebook(master)
        nb.pack(expand=1)
        self.Edit = Frame(master)
        self.Model = Frame(master)
        nb.add(self.Edit, text="Edit")
        nb.add(self.Model, text="Model")

class Colorpick:
    colorpick = 0
    def __init__(self, master):
        combobox_values = "\n".join(colorsdict.values())
        self.colorpick = ttk.Combobox(master, values=combobox_values)
        self.colorpick.current(0)
        self.colorpick.pack()

class Slider:
    slider = 0
    slidertext = 0
    def __init__(self, master):
        sv = StringVar()
        self.slider = Scale(master, from_=1, to=100, orient=HORIZONTAL, showvalue=0, length=500)
        self.slider.pack()
        self.textinit(master)
        self.slidertext.config(textvariable=sv)
        self.slider.config(variable=sv)

    def textinit(self, master):
        self.slidertext = Entry(master)
        self.slidertext.pack()

class RadioButt:
    rb = []
    Modes = ["Scipy", "Verlet", "Verlet-threading", "Verlet-multiprocessing", "Verlet-cython", "Verlet-opencl"]
    def __init__(self, master, plot, t):
        def sel():
            selection = v.get()
            Sun = [[0, 0, 0], [0, 0, 0], 1.99e30]
            Earth = [[-1.496e11, 0, 0], [0, 29.783e3, 0], 5.98e24]
            Moon = [[-1.496e11, -384467000, 0], [1020, 29.783e3, 0], 7.32e22]
            Mercury = [[-5.791e10, 0, 0], [0, 48e3, 0], 3.285e23]
            Mars = [[-2.279e11, 0, 0], [0, 24.1e3, 0], 6.39e23]
            time = 3.154e7
            Bodies = [Sun, Moon, Earth, Mercury, Mars]
            sol = VerletModule(Bodies, time, selection, t)
            PrintOrbit(plot, sol)

        v = StringVar()
        v.set(1)
        for text in self.Modes:
            self.rb.append(Radiobutton(master, text=text, variable=v, value=text, command=sel).pack(anchor=W))


#Хранит круги
class CircleList:
    circlelist = []
    def addcircle(self, _r, _x, _y, _color):
        circle = {'r': _r, 'x': _x, 'y': _y, 'color': _color}
        self.circlelist.append(circle)

def pend(y, t, m):
    N = len(m)
    G = constants.G
    dydt = []
    for i in range(N):
        dydt.append(y[i * 6 + 3])
        dydt.append(y[i * 6 + 4])
        dydt.append(y[i * 6 + 5])
        f1 = 0
        f2 = 0
        f3 = 0
        rv = y
        for j in range(N):
            if i != j:
                r = (np.linalg.norm([rv[j * 6] - rv[i * 6], rv[j * 6 + 1] - rv[i * 6 + 1], rv[j * 6 + 2] - rv[i * 6 + 2]])) ** 3
                tmp = G * m[j] / r
                f1 = f1 + (rv[j * 6] - rv[i * 6]) * tmp
                f2 = f2 + (rv[j * 6 + 1] - rv[i * 6 + 1]) * tmp
                f3 = f3 + (rv[j * 6 + 2] - rv[i * 6 + 2]) * tmp
        dydt.append(f1)
        dydt.append(f2)
        dydt.append(f3)
    return dydt

#Модуль для всех методов Верле
def VerletModule(Bodies, cycle_time, funcname, t):
    t.set("Computation time: ")
    start_time = time.time()
    if funcname == "Scipy":
        sol = ScipySolve(Bodies, cycle_time)
    if funcname == "Verlet":
        sol = Verlet(Bodies, cycle_time)
    if funcname == "Verlet-threading":
        sol = Verlet_threading(Bodies, cycle_time)
    if funcname == "Verlet-multiprocessing":
        sol = Verlet_multiprocessing(Bodies, cycle_time)
    t.set("Computation time: " + str(time.time() - start_time))
    return sol


#Вычисление положения тел в течение времени time. Params - массив, описывающий тела
def ScipySolve(params, time):
    N = len(params)
    t = np.linspace(0, time, 101)
    y0 = []
    m = []
    for i in range(N):
        y0.extend([params[i][0][0], params[i][0][1], params[i][0][2], params[i][1][0], params[i][1][1], params[i][1][2]] )
        m.append(params[i][2])
    sol = integrate.odeint(pend, y0, t, args=(m,), mxstep=5000000)
    sol1 = []
    for i in range(len(t)):
        sol1.append([])
        for j in range(N):
            sol1[i].append([])
            sol1[i][j].append(sol[i][j * 6])
            sol1[i][j].append(sol[i][j * 6 + 1])
            sol1[i][j].append(sol[i][j * 6 + 2])
            sol1[i][j].append(sol[i][j * 6 + 3])
            sol1[i][j].append(sol[i][j * 6 + 4])
            sol1[i][j].append(sol[i][j * 6 + 5])
    return sol1

#Алгоритм Верле для гравитационной задачи N тел
def Verlet(params, time):
    N = len(params)
    G = constants.G
    t = np.linspace(0, time, 101)
    dt = t[1] - t[0]
    sol = []
    y0 = []
    m = []
    for i in range(N):
        y0.append([params[i][0][0], params[i][0][1], params[i][0][2], params[i][1][0], params[i][1][1], params[i][1][2]])
        m.append(params[i][2])
    sol.append(y0)
    # print("Sol ", 0, sol[0])
    A = []
    for j in range(N):
        # print("A ", i, A)
        A.append([])
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
        A[j].extend([ai1, ai2, ai3])
    # print("A: ", A)
    for i in range(1, len(t)):
        # print("A ", i, A)
        iter = []
        sol.append([])
        for j in range(N):
            iter.append(sol[i - 1][j][0] + sol[i - 1][j][3] * dt + 0.5 * A[j][0] * dt ** 2)
            iter.append(sol[i - 1][j][1] + sol[i - 1][j][4] * dt + 0.5 * A[j][1] * dt ** 2)
            iter.append(sol[i - 1][j][2] + sol[i - 1][j][5] * dt + 0.5 * A[j][2] * dt ** 2)
        # print("A ", i, A)
        # print("iter ", i, iter)
        # print("sol[i - 1][0][0] ", i - 1, sol[i - 1][0])
        # print("sol[i - 1][0][3] ", i - 1, sol[i - 1][3])
        # print("0.5 * A[0][0] * dt ** 2 ", i - 1, 0.5 * A[0][0] * dt ** 2)
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
            sol[i].append([iter[j * 3], iter[j * 3 + 1], iter[j * 3 + 2], sol[i - 1][j][3] + 0.5 * (f1 + A[j][0]) * dt,
                       sol[i - 1][j][4] + 0.5 * (f2 + A[j][1]) * dt, sol[i - 1][j][5] + 0.5 * (f3 + A[j][2]) * dt])
        A = F
    return sol

#sync between threads
def Sync(events, controlevent, loopevent, isfreeevent):
    print("Sync")
    N = len(events)
    while(1):
        loopevent.wait()
        if(isfreeevent.is_set()):
            break
        for i in range(N):
            events[i].wait()
            events[i].clear()
            # print("Event number: ", i)
        controlevent.set()
        loopevent.clear()

def Verlet_threading_func(params, body_num, time, sol, y0, m, A, F, iter, event, controlevent, loopthread):
    N = len(params)
    G = constants.G
    tmp = range(N)
    A.append([])
    F.append([])
    iter.append([0, 0, 0])
    t = np.linspace(0, time, 101)
    dt = t[1] - t[0]
    y0[body_num] = [params[body_num][0][0], params[body_num][0][1], params[body_num][0][2], params[body_num][1][0], params[body_num][1][1], params[body_num][1][2]]
    m[body_num] = params[body_num][2]
    #sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    if (body_num == 0):
        sol.append(y0)
    #sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    ai1 = 0
    ai2 = 0
    ai3 = 0
    for k in range(N):
        if k != body_num:
            r = np.linalg.norm(
                [sol[0][k][0] - sol[0][body_num][0], sol[0][k][1] - sol[0][body_num][1],
                 sol[0][k][2] - sol[0][body_num][2]]) ** 3
            tmp = G * m[k] / r
            ai1 = ai1 + tmp * (sol[0][k][0] - sol[0][body_num][0])
            ai2 = ai2 + tmp * (sol[0][k][1] - sol[0][body_num][1])
            ai3 = ai3 + tmp * (sol[0][k][2] - sol[0][body_num][2])
    A[body_num] = [ai1, ai2, ai3]
    #sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    for i in range(1, len(t)):
        if(body_num == 0):
            sol.append([])
        # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        sol[i].append([])
        iter[body_num][0] = sol[i - 1][body_num][0] + sol[i - 1][body_num][3] * dt + 0.5 * A[body_num][0] * dt ** 2
        iter[body_num][1] = sol[i - 1][body_num][1] + sol[i - 1][body_num][4] * dt + 0.5 * A[body_num][1] * dt ** 2
        iter[body_num][2] = sol[i - 1][body_num][2] + sol[i - 1][body_num][5] * dt + 0.5 * A[body_num][2] * dt ** 2
        # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        f1 = 0
        f2 = 0
        f3 = 0
        for k in range(N):
            if k != body_num:
                r = np.linalg.norm([iter[k][0] - iter[body_num][0], iter[k][1] - iter[body_num][1],
                                    iter[k][2] - iter[body_num][2]]) ** 3
                tmp = G * m[k] / r
                f1 = f1 + tmp * (iter[k][0] - iter[body_num][0])
                f2 = f2 + tmp * (iter[k][1] - iter[body_num][1])
                f3 = f3 + tmp * (iter[k][2] - iter[body_num][2])
        F[body_num] = [f1, f2, f3]
        #sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        sol[i][body_num] = [iter[body_num][0], iter[body_num][1], iter[body_num][2], sol[i - 1][body_num][3] + 0.5 * (f1 + A[body_num][0]) * dt,
             sol[i - 1][body_num][4] + 0.5 * (f2 + A[body_num][1]) * dt, sol[i - 1][body_num][5] + 0.5 * (f3 + A[body_num][2]) * dt]
        # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        A[body_num] = F[body_num]
    return sol

#Алгоритм Верле с использованием модуля threading
def Verlet_threading(params, time):
    N = len(params)
    #synced lists
    e = []
    t = []
    sol = []
    y0 = []
    m = []
    A = []
    iter = []
    F = []
    for i in range(N):
        y0.append([])
        m.append([])
        e.append(threading.Event())
        # e[i].set()
    isfreeevent = threading.Event()
    controlevent = threading.Event()
    loopevent = threading.Event()
    controlthread = threading.Thread(target=Sync, args=(e, controlevent, loopevent, isfreeevent))
    controlthread.start()
    for i in range(N):
        t.append(threading.Thread(target=Verlet_threading_func, args=(params, i, time, sol, y0, m, A, F, iter, e[i], controlevent, loopevent)))
        t[i].start()
    for i in range(N):
        t[i].join()
    loopevent.set()
    isfreeevent.set()
    controlthread.join()
    return sol

def Verlet_multiprocessing_func(params, body_nums, process_num, time, event, controlevent, loopthread, sol, A, F, iter, y0, m):
    N = len(params)
    G = constants.G
    tmp = range(N)
    t = np.linspace(0, time, 101)
    dt = t[1] - t[0]
    for i in body_nums:
        y0[i * 6] = params[i][0][0]
        y0[i * 6 + 1] = params[i][0][1]
        y0[i * 6 + 2] = params[i][0][2]
        y0[i * 6 + 3] = params[i][1][0]
        y0[i * 6 + 4] = params[i][1][1]
        y0[i * 6 + 5] = params[i][1][2]
        m[i] = params[i][2]
    # sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    if (process_num == 0):
        for ind in range(len(y0)):
            sol[ind] = y0[ind]
    # sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    ai1 = 0
    ai2 = 0
    ai3 = 0
    for j in body_nums:
        for k in range(N):
            if k != j:
                r = np.linalg.norm(
                    [sol[k * 6] - sol[j * 6], sol[k * 6 + 1] - sol[j * 6 + 1],
                     sol[k * 6 + 2] - sol[j * 6 + 2]]) ** 3
                tmp = G * m[k] / r
                ai1 = ai1 + tmp * (sol[k * 6] - sol[j * 6])
                ai2 = ai2 + tmp * (sol[k * 6 + 1] - sol[j * 6 + 1])
                ai3 = ai3 + tmp * (sol[k * 6 + 2] - sol[j * 6 + 2])
        A[j * 3] = ai1
        A[j * 3 + 1] = ai2
        A[j * 3 + 2] = ai3
    # sync
    event.set()
    loopthread.set()
    controlevent.wait()
    controlevent.clear()
    for i in range(1, len(t)):
        print("i ", i)
        # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        for j in body_nums:
            iter[j * 3] = (sol[(i - 1) * N * 6 + j * 6] + sol[(i - 1) * N * 6 + j * 6 + 3] * dt + 0.5 * A[j * 3] * dt ** 2)
            iter[j * 3 + 1] = (sol[(i - 1) * N * 6 + j * 6 + 1] + sol[(i - 1) * N * 6 + j * 6 + 4] * dt + 0.5 * A[j * 3 + 1] * dt ** 2)
            iter[j * 3 + 2] = (sol[(i - 1) * N * 6 + j * 6 + 2] + sol[(i - 1) * N * 6 + j * 6 + 5] * dt + 0.5 * A[j * 3 + 2] * dt ** 2)
        # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        for j in body_nums:
            f1 = 0
            f2 = 0
            f3 = 0
            for k in range(N):
                if k != j:
                    r = np.linalg.norm([iter[k * 3] - iter[j * 3], iter[k * 3 + 1] - iter[j * 3 + 1],
                                        iter[k * 3 + 2] - iter[j * 3 + 2]]) ** 3
                    tmp = G * m[k] / r
                    f1 = f1 + tmp * (iter[k * 3] - iter[j * 3])
                    f2 = f2 + tmp * (iter[k * 3 + 1] - iter[j * 3 + 1])
                    f3 = f3 + tmp * (iter[k * 3 + 2] - iter[j * 3 + 2])
            F[j * 3] = f1
            F[j * 3 + 1] = f2
            F[j * 3 + 2] = f3
            # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        for j in body_nums:
            sol[i * N * 6 + j * 6] = iter[j * 3]
            sol[i * N * 6 + j * 6 + 1] = iter[j * 3 + 1]
            sol[i * N * 6 + j * 6 + 2] = iter[j * 3 + 2]
            sol[i * N * 6 + j * 6 + 3] = sol[(i - 1) * N * 6 + j * 6 + 3] + 0.5 * (F[j * 3] + A[j * 3]) * dt
            sol[i * N * 6 + j * 6 + 4] = sol[(i - 1) * N * 6 + j * 6 + 4] + 0.5 * (F[j * 3 + 1] + A[j * 3 + 1]) * dt
            sol[i * N * 6 + j * 6 + 5] = sol[(i - 1) * N * 6 + j * 6 + 5] + 0.5 * (F[j * 3 + 2] + A[j * 3 + 2]) * dt
            # sync
        event.set()
        loopthread.set()
        controlevent.wait()
        controlevent.clear()
        for j in body_nums:
            A[j * 3] = F[j * 3]
            A[j * 3 + 1] = F[j * 3 + 1]
            A[j * 3 + 2] = F[j * 3 + 2]
    print("Finished")
    return sol

#Алгоритм Верле с мультипроцессингом
def Verlet_multiprocessing(params, time):
    if __name__ == '__main__':
        count = multiprocessing.cpu_count() - 1
        # manager = Manager()
        t = np.linspace(0, time, 101)
        N = len(params)
        bod_in_proc = N // count
        exc = N % count
        curr_body = 0
        processes = []
        e = []
        sol = multiprocessing.Array("d", N * len(t) * 6)
        y0 = multiprocessing.Array("d", N * 6)
        m = multiprocessing.Array("d", N)
        A = multiprocessing.Array("d", N * 3)
        iter = multiprocessing.Array("d", N * 3)
        F = multiprocessing.Array("d", N * 3)
        # for i in range(N):
        #     y0.append([])
        #     m.append([])
        process_num = 0
        controlevent = multiprocessing.Event()
        loopevent = multiprocessing.Event()
        isfreeevent = multiprocessing.Event()
        if (count >= N):
            for i in range(N):
                e.append(multiprocessing.Event())
                processes.append(Process(target=Verlet_multiprocessing_func, args=(params, range(i, i + 1), process_num,
                                                                                   time, e[i], controlevent, loopevent, sol, A, F, iter, y0, m)))
                processes[i].start()
                process_num = process_num + 1
        else:
            for i in range(count):
                if exc > 0:
                    e.append(multiprocessing.Event())
                    processes.append(Process(target=Verlet_multiprocessing_func, args=(params, range(curr_body, curr_body + bod_in_proc + 1),
                                                                                       process_num, time, e[i], controlevent, loopevent, sol, A, F, iter, y0, m)))
                    processes[i].start()
                    process_num = process_num + 1
                    exc = exc - 1
                    curr_body = curr_body + bod_in_proc + 1
                else:
                    e.append(multiprocessing.Event())
                    processes.append(Process(target=Verlet_multiprocessing_func, args=(params, range(curr_body, curr_body + bod_in_proc),
                                                                                       process_num, time, e[i], controlevent, loopevent, sol, A, F, iter, y0, m)))
                    processes[i].start()
                    process_num = process_num + 1
                    curr_body = curr_body + bod_in_proc
        controlprocess = multiprocessing.Process(target=Sync, args=(e, controlevent, loopevent, isfreeevent))
        controlprocess.start()
        print("All set")
        for i in range(len(processes)):
            processes[i].join()
        isfreeevent.set()
        loopevent.set()
        controlprocess.join()
        print("Process finished")
        sol1 = []
        for i in range(len(t)):
            sol1.append([])
            for j in range(N):
                sol1[i].append([])
                sol1[i][j].append(sol[i * N * 6 + j * 6])
                sol1[i][j].append(sol[i * N * 6 + j * 6 + 1])
                sol1[i][j].append(sol[i * N * 6 + j * 6 + 2])
                sol1[i][j].append(sol[i * N * 6 + j * 6 + 3])
                sol1[i][j].append(sol[i * N * 6 + j * 6 + 4])
                sol1[i][j].append(sol[i * N * 6 + j * 6 + 5])
        return sol1


#Генерация К тел
def BodyGenerator(K):
    Bodies = []
    for i in range(K):
        Bodies.append([])
        if i == 0:
            Bodies[i].append([0, 0, 0])
            Bodies[i].append([0, 0, 0])
            m = random.uniform()


#Вычисление времени работы метода
def CompTime(methodname, N, M, Bodies):
    print("Tic-tock")
    avrtime = 0
    for i in range(M):
        start_time = time.time()
        VerletModule(Bodies, methodname)
        avrtime = avrtime + (time.time() - start_time)
    avrtime = avrtime / M
    return avrtime

#Отображение полученных орбит
def PrintOrbit(plot, sol):
    print("Start orbit visualiation")
    dt = len(sol)
    N = int(len(sol[0]))
    # print(sol[0], sol[1])
    # print("N: ", N)
    for i in range(dt):
        Axes.set_xlim(plot.subplot, -3e11, 3e11)
        Axes.set_ylim(plot.subplot, -3e11, 3e11)
        # ax.set_xlim(-2e11, 2e11)
        # ax.set_ylim(-2e11, 2e11)
        # ax.set_zlim(-2e11, 2e11)
        for j in range(N):
            circle = pyplot.Circle((sol[i][j][0], sol[i][j][1]), radius=2e9, color=colorsdict.get(j))
            plot.subplot.add_patch(circle)
            plot.canvas.show()
        time.sleep(0.01)
        plot.subplot.cla()
    print("Orbit done")

if __name__ == '__main__':
    root = Tk()
    CircleGUI(root)
    root.mainloop()
