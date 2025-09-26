#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:31:38 2025

@author: mollyfreeman
"""
import numpy as np
import matplotlib.pyplot as plt



"""Följande är svar på fråga F1"""
#F1a) 
#Vi ska rita riktiningsfältet för DE och lägg till den exakta lösningen till ekvationen


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
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

#Plotting the vector field F(x,y) = (-y,x)
# Define the vector field functions
# def u(x, y):
#     return -y

# def v(x, y):
#     return x


#Plotting riktningsfältet till dydx = x - y
#Vector field is: F(x,y) = (1,x-y)
def u(x, y):
    return 1

def v(x, y):
    return x-y

def main_F1a():
    # Call the function
    plot_quiver(x_range=(0, 1), y_range=(0, 1), u_func=u, v_func=v, density=20, scale=50, title="Riktningsfält")





#F1b) 
#Approximera metoden med Eulers metod framåt (skriv allmänt så koden kan användas igen) och steglängden h=0.1
#Spara initialdata och alla lösningsvärden i en vektor
#Plotta den numeriska lösningsvektorn som funktion av tiden



def dydt(t, y):
    return 1+t-y

def euler_forward_h(f, a, b, y0, h):

    n = int(np.abs(b-a)/h)
    print(n)
    # Diskretiseringsteg: Generate n+1 friddpunkter
    t = np.linspace(a, b, n+1)
    y = np.zeros(n+1)

    # Begynnelsevillkor
    y[0] = y0

    # Iterate med Eulers framåt. Formler EF.1
    # Modul 6 Del 2
    for k in range(n):
        y[k+1] = y[k] + h*dydt(t[k], y[k])

    return t, y

def get_table(tsol,ysol,h):
    
    yexakt = y_exakt(tsol)
    Ek_global = np.abs(ysol-yexakt)
    print("felet ek")
    print(Ek_global)
    
    tabell = {"Steg: tk": tsol,
              "yk": ysol
              }
              
    #Datatable
    #DT = dt.Frame(tabell)
    df = pd.DataFrame(tabell)
    
    print(df)
    
def y_exakt(t):
    return np.exp(-t)+t
    
    
def main_F1b():
    T = 1.2
    a = 0
    b = T
    y0 = 0
    h = 0.1
    # a)
    ta, ya = euler_forward_h(dydt, a, b, y0, h)
    #plotta_solution(ta,ya,h)
    get_table(ta, ya, h)
    
    #f = lambda t, y: -y      #Definera en lambda funktion (=inline function)
    sol = integrate.solve_ivp(dydt,[0,2],[0],t_eval=np.linspace(0,2,50))
    print(" ")
    print(sol.t)
    print(sol.y)   
    fig, ax = plt.subplots()
    ax.plot(sol.t,sol.y[0])    # sol.y är en matris, lösningen ges i första raden
    ax.set_xlabel('t',fontsize=14)
    ax.set_ylabel('y(t)',fontsize=14)
    ax.tick_params(labelsize=14)
    plt.show()

#F1c)
#Verifiera att felet blir ek = |yk(T)−yexakt(T)| ≈ 0.0188



#-----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga F2"""
#F2a)
#Gör en konvergensstudie genom att ha h och beräkna numerisk lösning med Eulers metod framåt
#Spara lösningen för varje h i y(T)

#F2b)
#Beräkna felen ek = |yk(T) − yexakt(T)| och verifiera att felen blir enligt facit

#F2c)
#Beräkna nogrannhetsordningen empiriskt och jämför 



#-----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga T1"""
#T1a)
#Gör en differentialekvation till en LCR krets genom att skriva om ekvation 2 till y´= F(t,y) 
#Liknande frågor kommer i elektroteknik

print("hej")

#Exempel: Om det är 3:e ordningens differentiaekvaiton (det finns åtminstånde en tredjederivata) så är det 3 stycken ekvationer men f(t,u1,u2,u3).
#f(t,u_vektor) där u_vektor är (u1,u2,u3).
#u_vektor' = (u1',u2',u3') = (u2, u3, y''') = (u2,u3, )
#u1 = y
#u1' = u2 = y'
#u2' = u3 = y''
#y''' = -0.5y*y''-(y')^2+sin(x) 
#y(0)=
#u(0)=(0,1,2)
#man byter ut
#man kan inte lösa högre ordningens differentialekvation. Rk4, Euler framåt för system, Euler bakår för system. 
#Kan man inte skriva om det på den formen så är det inte bra.
#Man måste kunna skriva om ett system av ekvationer.
#Den högsta ordningens DE på ena sidan och resten på andra sidan. 
#Döp om det! y = u1 som standard. y' = u2. uvektorn = (u1,u2). u' = fvektor(t,uvekotn). uvektor' = (u1'.u2')= (u2, u3)
#Givet att man hittade rätt svar, så 
#t0 = det som finns i begynnelsevillkorens insida. 




#T1b)
#t, y, R, L, C är inparametrar. Funktionen returnerar vektorn F(t, y). Man ska inte kunna använda ode-lösare solve_ivp

#T1c)
#Lös systemet med solve_ivp och metoden RK45 för dämpad  och odämpad svängning

#T1d)
#Dela tidsintervallet och plotta lösningen för varje värde på N. 

#T1e)
#För dämpad svänging, utför en konvergensstudie för Euler framåt, nogrannhetsordning empiriskt



#-----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga T1"""
#T2a)
#Temperaturfördelningen ska diskretiseras med centrala finita differenser för N=4.
#Skriv systemmatrisen och Hl med alla element. 

#T2b)
#Skriv samma men för generellt N. 

#T2c)
#Returnera systemmatrisen A och högerledet.

#T2d)
#Lös randvärdesproblemet med N=100. 

#T2e)
#Gör konvergensstudie med steglängdshalvering. 

#T2f)
#Testa att ändra randvillkoren och se om det ändras. 