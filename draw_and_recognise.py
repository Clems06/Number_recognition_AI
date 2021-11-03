#from number_recognition import *
import random
import math
import numpy as np
import jsonpickle
import atexit
import json
import keyboard
from tkinter import Tk,Canvas, font


with open("./save_net.json") as f:
    print("Starting importing net...")
    net = jsonpickle.decode(f.read())
print("Import net complete")

matrix = [0]*(28*28)
border, spacement = 50, 20


def erase_all(evt):
    global matrix
    matrix = [0]*(28*28)
    draw_table()


def click(evt):
    global border, spacement, matrix

    circle = [[  0,    0,  0,    0,    0],
              [  0,    0,  0.5, 0,    0],
              [  0,  0.5,    1,  0.5,  0],
              [  0,    0,  0.5, 0,    0],
              [  0,    0,  0,    0,    0 ]]
    x_block = (evt.x - border) // spacement
    y_block = (evt.y - border) // spacement
    for x in range(-2,3):
        for y in range(-2,3):
            if 0 <= x_block-x < 28 and 0 <= y_block-y < 28:
                matrix[(y_block-y)*28+x_block-x] += circle[x+2][y+2]
                matrix[(y_block - y) * 28 + x_block - x] = min(matrix[(y_block-y)*28+x_block-x],1)

    draw_table()


def draw_table():
    global border, spacement, matrix, net

    guess = net.calculateResult(matrix)
    confidence = max(guess)
    guess = guess.index(confidence)

    canvas.delete("all")
    for i in range(29):
        canvas.create_line(border + i*spacement, border, border+i*spacement, border+28*spacement,width=2,fill="black")
        canvas.create_line(border, border + i * spacement, border + 28 * spacement, border + i * spacement, width=2,fill="black")
    for x in range(28):
        for y in range(28):
            background=matrix[y * 28 + x]
            if background == 0:
                continue

            background = "#"+str(round((1-background)*255))*3
            canvas.create_rectangle(border+x*spacement,border+y*spacement,border+(x+1)*spacement,border+(y+1)*spacement, fill=background)

    Font = font.Font(size=10)
    canvas.create_text(350,border*2+28*spacement,text=str(guess)+"  Confidence: "+str(round(confidence*100))+"%", font=Font)


taille = 700
Fenetre = Tk()
Fenetre.geometry(str(taille)+"x"+str(taille))
canvas = Canvas(Fenetre,width=taille,height=taille,borderwidth=0,highlightthickness=0,bg="lightgray")
canvas.pack()
Fenetre.bind("<B1-Motion>",click)
Fenetre.bind("<Button-1>",click)
Fenetre.bind("<KeyPress-space>",erase_all)

Fenetre.after(100,draw_table)
Fenetre.mainloop()
