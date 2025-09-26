#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:47:21 2025

@author: arvidolin
"""


import numpy as np
import matplotlib.pyplot as plt

def plot_quiver(x_range, y_range, u_func, v_func, density=20, scale=1, title="Quiver Plot"):
    """
    Plots a quiver plot of a vector field defined by u_func and v_func.

    Parameters:
    - x_range: tuple (xmin, xmax)
    - y_range: tuple (ymin, ymax)
    - u_func, v_func: functions of (x, y) returning vector components
    - density: number of arrows along each axis
    - scale: scale factor for arrows
    - title: title of the plot
    """
    x = np.linspace(x_range[0], x_range[1], density)
    y = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x, y)
    
    U = u_func(X, Y)
    V = v_func(X, Y)
    
    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, scale=scale)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    
    # Plotting the exact solution
    t_exact = np.linspace(x_range[0], x_range[1], 100)
    y_exact = np.exp(-t_exact) + t_exact
    plt.plot(t_exact, y_exact, 'r-', linewidth=2, label="Exakt lösning")
    
    plt.legend()
    plt.show()

# Funktioner för riktningsfältet för dy/dt = 1 + t - y
def u(t, y):
    return 1

def v(t, y):
    return 1 + t - y

def main():
    T = 1.2
    # Anropa funktionen för att rita ut riktningsfältet och den exakta lösningen
    plot_quiver(x_range=(0, T), y_range=(0, 2), u_func=u, v_func=v, density=20, scale=50, title="Riktningsfält och exakt lösning för dy/dt = 1 + t - y")

if __name__ == "__main__":
    main()