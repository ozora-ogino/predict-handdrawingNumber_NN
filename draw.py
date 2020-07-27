import pygame
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from tkinter import *
import tkinter as tk
from tkinter import messagebox

class pixel():
    def __init__(self, x, y, width, height):
        self.x = x 
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def get_neighbors(self, g):
        #get the neighbors of each pixel in the grid, this is used fo drawing thicker lines.
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols=28

        #horizontal and vertical neighbors
        if i < cols - 1: #right
            self.neighbors.append(g.pixels[i+1][j])
        if i > 0: #left
            self.neighbors.append(g.pixels[i-1][j])
        if j < rows - 1: #up
            self.neighbors.append(g.pixels[i][j+1])
        if j > 0: #down
            self.neighbors.append(g.pixels[i][j-1])

        #Diagonal neighbors
        if j > 0 and i > 0: #top left
            self.neighbors.append(g.pixels[i-1][j-1])
        if j + 1 < rows and i > -1 and i-1 >0: #bottom left
            self.neighbors.append(g.pixels[i-1][j+1])
        if j -1 < rows and i < cols -1 and j -1 > 0: #top right
            self.neighbors.append(g.pixels[i+1][j-1])
        if j < rows-1 and i < cols-1: #bottom right
            self.neighbors.append(g.pixels[i+1][j+1])
    


class grid():
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row*col
        self.width = width
        self.height = height
        self.generatePixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)
    
    def generatePixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []

        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(pixel(x_gap *c, y_gap * r, x_gap, y_gap))


        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].get_neighbors(self)

    def clicked(self, pos):
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        li = self.pixels
        newMatrix = [[] for x in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li[i])):
                if li[i][j].color ==(255, 255, 255):
                    newMatrix[i].append(0)
                else:
                    newMatrix[i].append(1)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test / 255
        for row in range(28):
            for x in range(28):
                x_test[0][row][x] = newMatrix[row][x]

        return x_test[:1]


def guess(li):
    model = tf.keras.models.load_model('mnist.model')
    pred = model.predict(li)
    t = np.argmax(pred[0])
    print(f'I guess this number is {t}')

    window = tk.Tk()
    window.withdraw()
    messagebox.showinfo('Prediction', 'I guess this number is ' + str(t))
    window.destory()


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                guess(li)
                g.generatePixels()
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0,0,0)
                for n in clicked.neighbors:
                    n.color = (0, 0, 0)

            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255, 255, 255)
                except:
                    pass
        g.draw(win)
        pygame.display.update()


pygame.init()
width = 560
height = 560
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('Number Prediction')
g = grid(28, 28, width, height)
main()

pygame.quit()
quit()
