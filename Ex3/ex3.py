import sys
import numpy
import os
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tkinter import *
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfilename

class CircleGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Task 3")
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
            circle = pyplot.Circle((x, y), radius=radius, color=colour)
            plot.subplot.add_patch(circle)
            plot.canvas.show()

    def OpenFile(event):
        filename = askopenfilename(filetypes=[("XML files", "*.xml")])
        f = open(filename)

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
        Axes.set_xlim(self.subplot, -100, 100)
        Axes.set_ylim(self.subplot, -100, 100)

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

root = Tk()
CircleGUI(root)
root.mainloop()
