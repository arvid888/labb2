#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:31:38 2025

@author: mollyfreeman


Hejsan molly!
Hejsan Arvid! :D
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.integrate as integrate
import scipy.linalg
from scipy.sparse import diags #diags är en funktion


"""Följande är svar på fråga F1"""
# F1a)
# Vi ska rita riktiningsfältet för DE och lägg till den exakta lösningen till ekvationen

def console_clear():
    os.system('clear')

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

# Plotting the vector field F(x,y) = (-y,x)
# Define the vector field functions
# def u(x, y):
#     return -y

# def v(x, y):
#     return x


# Plotting riktningsfältet till dydx = x - y
# Vector field is: F(x,y) = (1,x-y)
def u(x, y):
    return 1


def v(x, y):
    return x-y


def main_F1a():
    # Call the function
    plot_quiver(x_range=(0, 1), y_range=(0, 1), u_func=u, v_func=v,
                density=20, scale=50, title="Riktningsfält")


# F1b)
# Approximera metoden med Eulers metod framåt (skriv allmänt så koden kan användas igen) och steglängden h=0.1
# Spara initialdata och alla lösningsvärden i en vektor
# Plotta den numeriska lösningsvektorn som funktion av tiden


def dydt(t, y):
    return 1+t-y


def euler_forward_h(f, a, b, y0, h):

    n = round(np.abs(b-a)/h)
    print(n)
    # Diskretiseringsteg: Generate n+1 friddpunkter
    t = np.linspace(a, b, n+1)
    y = np.zeros(n+1)

    # Begynnelsevillkor
    y[0] = y0

    # Iterate med Eulers framåt. Formler EF.1
    # Modul 6 Del 2
    for k in range(n):
        y[k+1] = y[k] + h*f(t[k], y[k])

    return t, y


def get_table(tsol, ysol, h):

    yexakt = y_exakt(tsol)
    Ek_global = np.abs(ysol-yexakt)
    print("felet ek")
    print(Ek_global)

    tabell = {"Steg: tk": tsol,
              "yk": ysol
              }

    # Datatable
    # DT = dt.Frame(tabell)
    df = pd.DataFrame(tabell)

    print(df)


def y_exakt(t):
    return np.exp(-t)+t


def main_F1b():
    a = 0
    b = 1.2
    y0 = 1
    h = 0.1
    # a)
    ta, ya = euler_forward_h(dydt, a, b, y0, h)
    print("ya")
    print(ya[-1])
    print("ta")
    print(ta[-1])
    yexakt = y_exakt(ta[-1])
    Ek = np.abs(ya[-1]-yexakt)
    print(Ek)
    # plotta_solution(ta,ya,h)
    # get_table(ta, ya, h)

    # f = lambda t, y: -y      #Definera en lambda funktion (=inline function)

    sol = integrate.solve_ivp(dydt, [0, 2], [0], t_eval=np.linspace(0, 2, 50))
    print(" ")
    print(sol.t)
    print(sol.y)
    fig, ax = plt.subplots()
    # sol.y är en matris, lösningen ges i första raden
    ax.plot(sol.t, sol.y[0])
    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel('y(t)', fontsize=14)
    ax.tick_params(labelsize=14)
    plt.show()


main_F1a()
main_F1b()

# F1c)
# Verifiera att felet blir ek = |yk(T)−yexakt(T)| ≈ 0.0188


# -----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga F2"""
# F2a)
# Gör en konvergensstudie genom att ha h och beräkna numerisk lösning med Eulers metod framåt
# Spara lösningen för varje h i y(T)

# h = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])


def main_f2():
    fel_lista = []
    h_lista = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    a = 0
    b = 1.2
    y0 = 1
    h = 0.1
    for h in h_lista:
        ta, ya = euler_forward_h(dydt, a, b, y0, h)
        yexakt = y_exakt(ta[-1])
        Ek = np.abs(ya[-1]-yexakt)
        fel_lista.append(Ek)
        print("för h = ", h, "så är lösningen", ya[-1], "och felet Ek = ", Ek)

    print(" \nuppgift f2c, nogranhetsordningarna är:")

    p_lista = []

    for i in range(len(fel_lista) - 1):
        ek_h = fel_lista[i]
        ek_h_halverad = fel_lista[i+1]
        p = np.log(ek_h / ek_h_halverad) / np.log(2)
        p_lista.append(p)
        print(p)

    # print(p_lista)


main_f2()


# F2b)
# Beräkna felen ek = |yk(T) − yexakt(T)| och verifiera att felen blir enligt facit

# F2c)
# Beräkna nogrannhetsordningen empiriskt och jämför


# -----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga T1"""
# T1a)
# Gör en differentialekvation till en LCR krets genom att skriva om ekvation 2 till y´= F(t,y)
# Liknande frågor kommer i elektroteknik

print("hej")

# Exempel: Om det är 3:e ordningens differentiaekvaiton (det finns åtminstånde en tredjederivata) så är det 3 stycken ekvationer men f(t,u1,u2,u3).
# f(t,u_vektor) där u_vektor är (u1,u2,u3).
# u_vektor' = (u1',u2',u3') = (u2, u3, y''') = (u2,u3, )
# u1 = y
# u1' = u2 = y'
# u2' = u3 = y''
# y''' = -0.5y*y''-(y')^2+sin(x)
# y(0)=
# u(0)=(0,1,2)
# man byter ut
# man kan inte lösa högre ordningens differentialekvation. Rk4, Euler framåt för system, Euler bakår för system.
# Kan man inte skriva om det på den formen så är det inte bra.
# Man måste kunna skriva om ett system av ekvationer.
# Den högsta ordningens DE på ena sidan och resten på andra sidan.
# Döp om det! y = u1 som standard. y' = u2. uvektorn = (u1,u2). u' = fvektor(t,uvekotn). uvektor' = (u1'.u2')= (u2, u3)
# Givet att man hittade rätt svar, så
# t0 = det som finns i begynnelsevillkorens insida.


# T1b)

def diffekvation_2(t, y, R, L, C):
    q = y[0]  # detta är för solve_ivp
    i = y[1]  # första derivatan

    q_prim = i
    i_prim = -(R/L)*i-(1/L*C)*q

    return np.array([q_prim, i_prim])
# t, y, R, L, C är inparametrar. Funktionen returnerar vektorn F(t, y). Man ska inte kunna använda ode-lösare solve_ivp


def my_ode(t, y):
    """
    Definierar den ordinära differentialekvationen dy/dt = -2y.
    """
    return -2 * y


# 2. Definiera tidsintervall och initialvillkor
t_span = (0, 5)
y0 = [1]

# 3. Löser ekvationen med solve_ivp
sol = integrate.solve_ivp(my_ode, t_span, y0, dense_output=True)


def lös_ODE(R, L, C, t_span, Q0):

    y0 = [Q0, 0]

    F = integrate.solve_ivp(diffekvation_2, t_span, y0, args=(R, L, C))
    return F


# T1c)
# Lös systemet med solve_ivp och metoden RK45 för dämpad  och odämpad svängning


def main_T1_c():

    t_span = [0, 20]
    Q0 = 1
    L = 2
    C = 0.5
    R_dämpad = 1
    R_odämpad = 0
    lösning_dämpad = lös_ODE(R_dämpad, L, C, t_span, Q0)
    lösning_odämpad = lös_ODE(R_odämpad, L, C, t_span, Q0)
    print('lösningen för dämpad är ', lösning_dämpad)
    print('lösningen för odämpad är ', lösning_odämpad)


main_T1_c()

# T1d)
# Dela tidsintervallet och plotta lösningen för varje värde på N.


def euler_system_forward_h(F, t0, tend, U0, h,R,L,C):
    """ 
    Implements Euler's method forward for a system of ODEs.

    Parameters:
        F       : function(t, U) → dU/dt (returns numpy array)
        t0      : initial time
        tend    : Final time
        U0      : initial state (numpy array)
        h       : step size

    Returns:
        t_values : numpy array of time points
        y_values : numpy array of state values (n_steps+1 x len(y0))
    """

    n_steps = int(np.abs(tend-t0)/h)
    y0 = np.array(U0, dtype=float)
    t_values = np.zeros(n_steps+1)
    y_values = np.zeros((n_steps+1, len(y0)))

    t_values[0] = t0
    y_values[0] = U0

    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * F(t_values[i], y_values[i],R,L,C)
        t_values[i+1] = t_values[i] + h

    return t_values, y_values


def plotta_T1_d(x_värden, y_värden,N,h,exakt_lösning):
    
    fig, ax = plt.subplots()
    # sol.y är en matris, lösningen ges i första raden
    ax.plot(x_värden, y_värden)
    ax.plot(exakt_lösning.t, exakt_lösning.y[0], label='RK45 (Exakt)', linestyle='--')
    ax.set_xlabel('t för N = ' +str(N) + ' och h = ' + str(h), fontsize=14)
    ax.set_ylabel('y(t)', fontsize=14)
    ax.tick_params(labelsize=14)
    plt.show()
 

def T1_d():
    N_lista = np.array([20,40,80,160])
    t_intervall = [0,20]
    t = t_intervall[1] - t_intervall[0]
    t0 = t_intervall[0]
    tend = t_intervall[1]
    U0 = np.array([1,0])
    R = 1
    L = 2
    C = 0.5
    for N in N_lista:
        h = t/N
        lösning_euler = euler_system_forward_h(diffekvation_2, t0, tend, U0, h,R,L,C)
        #print("för N = ", N, "blir steglängden h= ", h, "och lösningen blir ", lösning_euler)
        exakt_lösning = lös_ODE(R, L, C, t_intervall, U0[0])
        plotta_T1_d(lösning_euler[0], lösning_euler[1],N,h,exakt_lösning)
    
    
def T1_e():
    N_lista = np.array([80,160,320,640])
    t_intervall = [0,20]
    t = t_intervall[1] - t_intervall[0]
    t0 = t_intervall[0]
    tend = t_intervall[1]
    U0 = np.array([1,0])
    R = 1
    L = 2
    C = 0.5
    föregående_fel = 0
    for N in N_lista:
        h = t/N
        lösning_euler = euler_system_forward_h(diffekvation_2, t0, tend, U0, h,R,L,C)
        exakt_lösning = lös_ODE(R, L, C, t_intervall, U0[0])
        exakta_y_värden = exakt_lösning.y
        felet = np.abs(lösning_euler[1][-1][0] - exakta_y_värden[0][-1]) #eftersom lösning_euler är en tupel
        print("felet för N = ", N,"och h = ", h, "är Ek = ", felet)
        
        
         
        if föregående_fel !=0:
            nogrannhetsordning = np.log2(np.abs(föregående_fel / felet))
            print("nogrannhetsordning = ", nogrannhetsordning)
        
        föregående_fel = felet
T1_d()   
T1_e()

    


    
# T1e)
# För dämpad svänging, utför en konvergensstudie för Euler framåt, nogrannhetsordning empiriskt

































# -----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga T1"""
# T2a, b och c) KLAR

def q(x): #Enligt uppgiften
    return 50 * (x**3) * np.log(x + 1)

#Variabler
N = 4 #här kan man ändra, tex 4, 100 osv enligt uppgiften
k = 2 #konstant enligt uppgiften
TL = 2 #ska vara 2 enligt uppgiften, men kan ändras enligt uppgiften
TR = 2 #ska vara 2 enligt uppgiften, men kan ändras enligt uppgiften
L = 1 #konstant enligt uppgiften

def stavens_temperatur(N, q, k, TL, TR, L): 
    #N = Antal delintervall, dvs N+1 punkter finns totalt men vi ska räkna ut N-1 punkter 
    #q = funktionen som ska vara q(x) enligt uppgiften
    #k = stavens värmeledningsförmåga, en skalär
    #TL = randvillkor vid x=0, 
    #TR = randvillkor vid x=L
    #L = längd på staven
    h = (L - 0) / N #steglängden ((b-a)/N ) = ( (slutpunkt - startpunkt)/ N antal delintervall ) är 0.25 om N=4
   
    #Nu ska A skrivas
    mittersta_diagnonalen = (-2*k/h**2) *np.eye(N-1) #(-2k / h**2) på mittersta diagonalen, 
    nedre_diagnonalen = (k/h**2) * np.diag(np.ones((N-2), dtype=None),-1) #https://numpy.org/doc/2.2/reference/generated/numpy.ones.html
    ovre_diagnonalen = (k/h**2)* np.diag(np.ones((N-2), dtype=None),1) 
    A = np.round(mittersta_diagnonalen + nedre_diagnonalen + ovre_diagnonalen) #A har 3 diagonaler, i formen av (N-1)*(N-1)
    print("A:\n", A)
    
    #Nu ska x_vektorn skrivas
    x_j = np.linspace(0+h, L-h, N-1) 
    #x_j värden skrivs i en lista, där indexet j = [1, 2, 3, …, (N-1))]
    #x_j elementen står för punkterna där vi ska approximera temperaturen
    #np.linspace(startpunkt, slutpunkt, antal x_j punkter som vi ska räkna ut)
    #https://numpy.org/devdocs//reference/generated/numpy.linspace.html 
    print("x värden:\n", x_j)
    
    #Nu ska b_vektorn skrivas
    b = q(x_j).astype(float)
    #Högerledsvektorn b har längden (N-1)
    b[0] -= (k / h**2) * TL 
    b[-1] -= (k / h**2) * TR
    print("b värden:\n", b)
    
    #Lös ut T inre och 
    T_ej_med_randvillkor = np.linalg.solve(A,b)

    x = np.linspace(0, L, N+1) #alla x_värden, dvs randvärderna är inkluderade
    T = np.concatenate(([TL],T_ej_med_randvillkor,[TR])) #alla T_värden, dvs randvärderna är inkluderade
    
    return A, x, b, T 

#printa temperaturerna i en lista
A, x, b, T = stavens_temperatur(N, q, k, TL, TR, L)
print("Stavens temperatur beräknat på",N+1,"stycken punkter är\n", T)

# T2d) KLAR
#Få fram A, x och b för N=100 istället.
A_T2d, x_T2d, b_T2d, T_T2d = stavens_temperatur(100, q, k, TL, TR, L)
#A_T2d systemmatrisen 99*99 för de inre punkterna
#x_T2d alla x koordinaterna där randvärderna är inkluderade
#b_T2d b för de inre punkterna
#T_T2d temperaturen där randvärderna är inkluderade

#Skriv funktionen som plottar temperaturfördelningen
def plotta_stavens_temperatur(x,T): 
    #x = vad som ska vara på x-axeln
    #T = vad som ska vara på y-axeln (som ska vara temperaturen)
    plt.plot(x, T, '-o', markersize=2.5) #markersize ändrar punkternas storlek
    plt.title("Stavens temperaturfördelning (N=100)")
    plt.xlabel("x")
    plt.ylabel("T(x), temperaturen")
    plt.grid(True) #Visar hjälplinjerna

#kalla på funktionen som plottar
plotta_stavens_temperatur(x_T2d,T_T2d)

#Printa vad temperaturen uppskattas vara i x=0.2
for idx, element in enumerate(x_T2d):
    #https://stackoverflow.com/questions/522563/how-can-i-access-the-index-value-in-a-for-loop
    if element == 0.2:
        print("Stavens temperatur i x=0.2 är", T_T2d[idx])

# T2e)

# Gör konvergensstudie med steglängdshalvering.

#Antal delintervall i lista, börja på N=50
N = [50, 100, 200, 400, 800, 1600]

T_jamforelsevarde_noll_sju_avrundat= 1.6379544


# T2f)
# Testa att ändra randvillkoren och se om det ändras.






