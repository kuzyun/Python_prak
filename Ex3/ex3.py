import sys
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
import mpl_toolkits.mplot3d as a3

colorsdict = {0: 'Red', 1: 'Blue', 2: 'Green', 3: 'Yellow'}
class CircleGUI:
    circles = 0
    def __init__(self, master):
        self.master = master
        self.master.title("Task 3")
        self.circles = CircleList
        plot = PlotArea(self.master)
        buttons = Buttons(master, plot)
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
    def __init__(self, master, plot):
        def sel():
            selection = v.get()
            Sun = [[0, 0, 0], [0, 0, 0], 1.99e30]
            Earth = [[-1.496e110, 0, 0], [0, -29.783e3, 0], 5.98e24]
            Moon = [[-1.496e11, -384467000, 0], [1020, -29.783e3, 0], 7.32e22]
            time = 3.154e7
            # Sun = [[0, 0], [0, 0], 10]
            # Earth = [[10, 0], [0, 2], 2]
            # Moon = [[9, 0], [0, -1], 1]
            # time = 100
            Bodies = [Sun, Earth, Moon]
            if selection == "Scipy":
                sol = ScipySolve(Bodies, time)
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
    return sol

#Алгоритм Верле для гравитационной задачи N тел
def Verlet(params, time):
    N = len(params)
    G = constants.G
    t = np.linspace(0, time, 101)
    dt = t[1] - t[0]
    sol = []
    y0 = []
    m = []
    sol.append([])
    for i in range(N):
        y0.extend([params[i][0][0], params[i][0][1], params[i][0][2], params[i][1][0], params[i][1][1], params[i][1][2]])
        m.append(params[i][2])
    sol[0].extend(y0)
    ai1 = 0
    ai2 = 0
    ai3 = 0
    A = []
    for j in range(N):
        A.append([])
        for k in range(N):
            if k != j:
                r = np.linalg.norm(
                    [sol[0][k * 6] - sol[0][j * 6], sol[0][k * 6 + 1] - sol[0][j * 6 + 1], sol[0][k * 6 + 2] - sol[0][j * 6 + 2]]) ** 3
                tmp = G * m[k] / r
                ai1 = ai1 + tmp * (sol[0][k * 6] - sol[0][j * 6])
                ai2 = ai2 + tmp * (sol[0][k * 6 + 1] - sol[0][j * 6 + 1])
                ai3 = ai3 + tmp * (sol[0][k * 6 + 2] - sol[0][j * 6 + 2])
        A[j].extend([ai1, ai2, ai3])
    for i in range(1, len(t)):
        iter = []
        sol.append([])
        for j in range(N):
            iter.append(sol[i - 1][j * 6] + sol[i - 1][j * 6 + 3] * dt + 0.5 * A[j][0] * dt ** 2)
            iter.append(sol[i - 1][j * 6 + 1] + sol[i - 1][j * 6 + 4] * dt + 0.5 * A[j][1] * dt ** 2)
            iter.append(sol[i - 1][j * 6 + 2] + sol[i - 1][j * 6 + 5] * dt + 0.5 * A[j][2] * dt ** 2)
        f1 = 0
        f2 = 0
        f3 = 0
        F = []
        for j in range(N):
            F.append([])
            for k in range(N):
                if k != j:
                    r = np.linalg.norm([iter[k * 3] - iter[j * 3], iter[k * 3 + 1] - iter[j * 3 + 1], iter[k * 3 + 2] - iter[j * 3 + 2]]) ** 3
                    tmp = G * m[k] / r
                    f1 = f1 + tmp * (iter[k * 3] - iter[j * 3])
                    f2 = f2 + tmp * (iter[k * 3 + 1] - iter[j * 3 + 1])
                    f3 = f3 + tmp * (iter[k * 3 + 2] - iter[j * 3 + 2])
            F[j].extend([f1, f2, f3])
            sol[i].extend([iter[j * 3], iter[j * 3 + 1], iter[j * 3 + 2], sol[i - 1][j * 6 + 3] + 0.5 * (f1 + A[j][0]) * dt,
                       sol[i - 1][j * 6 + 4] + 0.5 * (f2 + A[j][1]) * dt, sol[i - 1][j * 6 + 5] + 0.5 * (f3 + A[j][2]) * dt])
        A = F
    return sol

def Verlet_threading_func(params, body_num, time, q, event_for_wait, event_for_set):
    N = len(params)
    G = constants.G
    t = np.linspace(0, time, 101)
    dt = t[1] - t[0]
    sol = []
    y0 = []
    m = []
    sol.append([])
    for i in range(N):
        y0.extend([params[i][0][0], params[i][0][1], params[i][0][2], params[i][1][0], params[i][1][1], params[i][1][2]])
        m.append(params[i][2])
    sol[0].extend(y0)
    ai1 = 0
    ai2 = 0
    ai3 = 0
    A = []
    # for j in range(N):
    event_for_wait.wait()
    event_for_wait.clear()
    for k in range(N):
        if k != body_num:
            r = np.linalg.norm(
                [sol[0][k * 6] - sol[0][body_num * 6], sol[0][k * 6 + 1] - sol[0][body_num * 6 + 1],
                 sol[0][k * 6 + 2] - sol[0][body_num * 6 + 2]]) ** 3
            tmp = G * m[k] / r
            ai1 = ai1 + tmp * (sol[0][k * 6] - sol[0][body_num * 6])
            ai2 = ai2 + tmp * (sol[0][k * 6 + 1] - sol[0][body_num * 6 + 1])
            ai3 = ai3 + tmp * (sol[0][k * 6 + 2] - sol[0][body_num * 6 + 2])
    q.put([ai1, ai2, ai3])
    event_for_set.set()
    while not q.empty():
        A.append(q.get())
    print(A, body_num)
    for i in range(1, len(t)):
        iter = []
        sol.append([])
        # for j in range(N):
        event_for_wait.wait()
        event_for_wait.clear()
        iter.append(sol[i - 1][body_num * 6] + sol[i - 1][body_num * 6 + 3] * dt + 0.5 * A[body_num][0] * dt ** 2)
        iter.append(sol[i - 1][body_num * 6 + 1] + sol[i - 1][body_num * 6 + 4] * dt + 0.5 * A[body_num][1] * dt ** 2)
        iter.append(sol[i - 1][body_num * 6 + 2] + sol[i - 1][body_num * 6 + 5] * dt + 0.5 * A[body_num][2] * dt ** 2)
        q.put(iter)
        event_for_set.set()
        iter = []
        while not q.empty():
            iter.append(q.get())
        # print(iter)
        f1 = 0
        f2 = 0
        f3 = 0
        F = []
        for j in range(N):
            F.append([])
            for k in range(N):
                if k != j:
                    r = np.linalg.norm([iter[k * 3] - iter[j * 3], iter[k * 3 + 1] - iter[j * 3 + 1],
                                        iter[k * 3 + 2] - iter[j * 3 + 2]]) ** 3
                    tmp = G * m[k] / r
                    f1 = f1 + tmp * (iter[k * 3] - iter[j * 3])
                    f2 = f2 + tmp * (iter[k * 3 + 1] - iter[j * 3 + 1])
                    f3 = f3 + tmp * (iter[k * 3 + 2] - iter[j * 3 + 2])
            F[j].extend([f1, f2, f3])
            sol[i].extend(
                [iter[j * 3], iter[j * 3 + 1], iter[j * 3 + 2], sol[i - 1][j * 6 + 3] + 0.5 * (f1 + A[j][0]) * dt,
                 sol[i - 1][j * 6 + 4] + 0.5 * (f2 + A[j][1]) * dt, sol[i - 1][j * 6 + 5] + 0.5 * (f3 + A[j][2]) * dt])
        A = F
    return sol

#Алгоритм Верле с использованием модуля threading
def Verlet_threading(params, time):
    N = len(params)
    q = Queue()
    e1 = threading.Event()
    e2 = threading.Event()
    e3 = threading.Event()
    t1 = threading.Thread(target=Verlet_threading_func, args=(params, 0, time, q, e1, e2))
    t2 = threading.Thread(target=Verlet_threading_func, args=(params, 1, time, q, e2, e3))
    t3 = threading.Thread(target=Verlet_threading_func, args=(params, 2, time, q, e3, e1))
    t1.start()
    t2.start()
    t3.start()
    e1.set()
    res = q.get()



def PrintOrbit(plot, sol):
    dt = len(sol)
    N = int(len(sol[0]) / 6)
    print(sol[0], sol[1])
    for i in range(dt):
        Axes.set_xlim(plot.subplot, -2e11, 2e11)
        Axes.set_ylim(plot.subplot, -2e11, 2e11)
        # ax.set_xlim(-2e11, 2e11)
        # ax.set_ylim(-2e11, 2e11)
        # ax.set_zlim(-2e11, 2e11)
        for j in range(N):
            circle = pyplot.Circle((sol[i][j * 6], sol[i][j * 6 + 1]), radius=2e9, color=colorsdict.get(j))
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
