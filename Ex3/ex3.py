import sys
import numpy as np
import os
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename
import xml.etree.ElementTree as ET
from scipy import constants
from scipy import integrate
import scipy
import time

class CircleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Task 3")

        circles = CircleList

        plot = PlotArea(self.master)
        buttons = Buttons(master, plot)
        tabs = Tabs(master)
        labels = AxesLabels(tabs.Edit)
        plot.canvas.mpl_connect('motion_notify_event', lambda event: OnMove(event, coord=labels))
        plot.canvas.mpl_connect('axes_leave_event', lambda event: OnLeave(event, coord=labels))
        colorpick = Colorpick(tabs.Edit)
        slider = Slider(tabs.Edit)
        plot.canvas.mpl_connect('button_press_event', lambda event: OnClick(event))
        openbutton = Button(master, text="Open", command=self.OpenFile)
        openbutton.pack(fill=X)
        savebutton = Button(master, text="Save", command=self.SaveFile)
        savebutton.pack(fill=X)
        rb = RadioButt(tabs.Model, plot)

        def OnMove(event, coord):
            x, y = event.xdata, event.ydata
            coord.xcoord.config(text=x)
            coord.ycoord.config(text=y)

        def OnLeave(event, coord):
            coord.xcoord.config(text="")
            coord.ycoord.config(text="")

        def OnClick(event):
            # plot.subplot.cla()
            x, y = labels.xcoord.cget("text"), labels.ycoord.cget("text")
            radius = slider.slider.get()
            colour = colorpick.colorpick.get()
            circles.addcircle(circles, radius, x, y, colour)
            circle = pyplot.Circle((x, y), radius=radius, color=colour)
            plot.subplot.add_patch(circle)
            plot.canvas.show()

    def OpenFile(event):
        filename = askopenfilename(filetypes=[("XML files", "*.xml")])
        tree = ET.parse(filename)
        root = tree.getroot()
        # print(root)



    def SaveFile(event):
        filename = asksaveasfilename(filetypes=[("XML files", "*.xml")])
        f = open(filename, 'w')

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
        self.colorpick = ttk.Combobox(master, values=['Red', 'Green', 'Blue', 'Yellow'])
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
    def __init__(self, master, plot):
        def sel():
            selection = v.get()
            Sun = [[0, 0], [0, 0], 1.989e30]
            Earth = [[149.6e9, 0], [0, 3e4], 5.97e24]
            Moon = [[149.984e9, 0], [0, 1020], 7.36e22]
            time = 3.154e7
            # Sun = [[0, 0], [0, 0], 10]
            # Earth = [[10, 0], [0, 2], 2]
            # Moon = [[9, 0], [0, -1], 1]
            # time = 100
            Bodies = [Sun, Earth, Moon]
            if selection == "Scipy":
                sol = ScipySolve(Bodies, time)
                print(sol[0])
                PrintOrbit(plot, sol)
            if selection == "Verlet":
                sol = Verlet(Bodies, time)
                PrintOrbit(plot, sol)
            if selection == "Verlet-threading":
                sol = Verlet_threading(Bodies, time)
                PrintOrbit(plot, sol)

        v = StringVar()
        v.set(1)
        for text in self.Modes:
            self.rb.append(Radiobutton(master, text=text, variable=v, value=text, command=sel).pack(anchor=W))


#Хранит круги
class CircleList:
    circlelist = []
    def addcircle(self, _r, _x, _y, _color):
        circle = {'r':_r, 'x': _x, 'y': _y, 'color': _color}
        self.circlelist.append(circle)

def pend(y, t, m):
    N = len(m)
    G = constants.G
    dydt = []
    rv = y
    for i in range(N):
        dydt.append(rv[i * 4 + 2])
        dydt.append(rv[i * 4 + 3])
        f1 = 0
        f2 = 0
        for j in range(N):
            if i != j:
                f1 = f1 + m[j] * ((rv[j * 4] - rv[i * 4]) / (np.linalg.norm([rv[j * 4] - rv[i * 4], rv[j * 4 + 1] - rv[i * 4 + 1]])) ** 3)
                f2 = f2 + m[j] * ((rv[j * 4 + 1] - rv[i * 4 + 1]) / (np.linalg.norm([rv[j * 4] - rv[i * 4], rv[j * 4 + 1] - rv[i * 4 + 1]])) ** 3)
        dydt.append(G * f1)
        dydt.append(G * f2)
    return dydt

#Вычисление положения тел в течение времени time. Params - массив, описывающий тела
def ScipySolve(params, time):
    N = len(params)
    t = np.linspace(0, time, 101)
    y0 = []
    m = []
    for i in range(N):
        y0.extend([params[i][0][0], params[i][0][1], params[i][1][0], params[i][1][1]])
        m.append(params[i][2])
    sol = integrate.odeint(pend, y0, t, args=(m,), mxstep=5000000)
    return sol

#Алгоритм Верле для гравитационной задачи N тел
def Verlet(params, time):
    N = len(params)
    G = constants.G
    t = np.linspace(0, time, 11)
    dt = t[1] - t[0]
    sol = []
    y0 = []
    m = []
    sol.append([])
    for i in range(N):
        y0.extend([params[i][0][0], params[i][0][1], params[i][1][0], params[i][1][1]])
        m.append(params[i][2])
    sol[0].extend(y0)
    f1 = 0
    f2 = 0
    for j in range(N):
        for k in range(N):
            if k != j:
                f1 = f1 + m[k] * ((sol[0][k * 4] - sol[0][j * 4]) / (
                    np.linalg.norm(
                        [sol[0][k * 4] - sol[0][j * 4], sol[0][k * 4 + 1] - sol[0][j * 4 + 1]])) ** 3)
                f2 = f2 + m[k] * ((sol[0][k * 4 + 1] - sol[0][j * 4 + 1]) / (
                    np.linalg.norm(
                        [sol[0][k * 4] - sol[0][j * 4], sol[0][k * 4 + 1] - sol[0][j * 4 + 1]])) ** 3)
    ai1 = f1
    ai2 = f2
    for i in range(1, len(t)):
        iter = []
        sol.append([])
        for j in range(N):
            iter.append(sol[i - 1][j * 4] + sol[i - 1][j * 4 + 2] * dt + 0.5 * ai1 * dt * dt)
            iter.append(sol[i - 1][j * 4 + 1] + sol[i - 1][j * 4 + 3] * dt + 0.5 * ai2 * dt * dt)
        f1 = 0
        f2 = 0
        for j in range(N):
            for k in range(N):
                if k != j:
                    f1 = f1 + m[k] * ((iter[k * 2] - iter[j * 2]) / (
                    np.linalg.norm([iter[k * 2] - iter[j * 2], iter[k * 2 + 1] - iter[j * 2 + 1]])) ** 3)
                    f2 = f2 + m[k] * ((iter[k * 2 + 1] - iter[j * 2 + 1]) / (
                    np.linalg.norm([iter[k * 2] - iter[j * 2], iter[k * 2 + 1] - iter[j * 2 + 1]])) ** 3)
            sol[i].extend([iter[j * 2], iter[j * 2 + 1], sol[i - 1][j * 4 + 2] + 0.5 * (f1 + ai1) * dt, sol[i - 1][j * 4 + 3] + 0.5 * (f2 + ai2) * dt])
        ai1 = f1
        ai2 = f2
    return sol

def Verlet_threading(params, time):
    N = len(params)

def PrintOrbit(plot, sol):
    dt = len(sol)
    N = int(len(sol[0]) / 4)
    Colors = {1:'Red', 2:'Blue', 3:'Green', 4:'Yellow'}
    for i in range(dt):
        Axes.set_xlim(plot.subplot, -2e11, 2e11)
        Axes.set_ylim(plot.subplot, -2e11, 2e11)
        # Axes.set_xlim(plot.subplot, -20, 20)
        # Axes.set_ylim(plot.subplot, -20, 20)
        for j in range(N):
            circle = pyplot.Circle((sol[i][j * N], sol[i][j * N + 1]), radius=2e9, color=Colors.get(j))
            plot.subplot.add_patch(circle)
            plot.canvas.show()
        time.sleep(0.01)
        plot.subplot.cla()

#Координаты [x, y] * 10^9, скорость[vx, vy] * 10^3, масса * 10^22
# Sun = [[0, 0], [0, 0], 198900000]
# Earth = [[149.6, 0], [0, 30], 597]
# Moon = [[149.984, 0], [0, 1.02], 7.36]

# Sun = [[0, 0], [0, 0], 10]
# Earth = [[10, 0], [0, 2], 2]
# Moon = [[9, 0], [0, -1], 1]
# Bodies = [Sun, Earth, Moon]
# sol = ScipySolve(Bodies, 10)
# vec1 = [1, 1]
# vec2 = [2, 0]
# res = np.linalg.norm([])
# print(res)

root = Tk()
CircleGUI(root)
root.mainloop()
