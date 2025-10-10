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
#import scipy.linalg
#from scipy.sparse import diags #diags är en funktion


"""Följande är svar på fråga F1"""
# F1a)

#Skapar en funktion för att rensa konsolen
def console_clear():
    os.system('clear')

#skapa den exakta lösningen
def y_exakt(t):
    return np.exp(-t)+t

#differentialekvationen y' = dy/dt = 1+t-y = f(t,y) enligt uppgiften (tidigare v)
def f(t,y):
    return 1+t-y

#t' = dt/dt = 1 = t_prim(t,y). (tidigare u)
def t_prim(t, y):
    return 1

#Skapa en konstant T eftersom t ∈ [0, T]
T = 1.2

#definiera axlarna på plottens start och slutpunkt
t_start_slut=(0, 2*T)
y_start_slut=(0, 2*T)

#Skapa ett riktningsfält (plott_quiver = plotta_riktningsfalt) som ges av F(t,y) = (1,f(t,y))
def plotta_riktningsfalt(t_start_slut, y_start_slut, t_prim, f, tathet=20, scale = 50, titel = "Riktningsfält"):
    #(x_range, y_range, u_func, v_func, density=20, scale=1, title="Quiver Plot" ändras)
    #t_start_slut = tupel (t_min, t_max)
    #y_start_slut = tupel (y_min, y_max)
    #t_prim = t'
    #f = y'
    #tathet = 20 = antal pilar per axel
    #scale = 1 = pilarnas skalfaktor, desto högre siffra desto kortare
    #title = "..." titeln
    
    #skapa ett rutnät med punkter
    t_axel = np.linspace(t_start_slut[0], t_start_slut[-1], tathet)
    y_axel = np.linspace(y_start_slut[0], y_start_slut[-1], tathet)
    #(starpunkt, slutpunk, antal punkter)
    t, y = np.meshgrid(t_axel, y_axel)
    #t_axel vektorn som består av massa punkter
    #y_axel består av massa punkter
    #np.meshgrid skapar en matris av dessa
    
    #få fram riktning per axel
    U = t_prim(t, y) #pekar i t-led
    V = f(t, y) #pekar i y-led

    #plotta figuren
    plt.figure(figsize=(6, 6))
    plt.quiver(t, y, U, V, scale=None)
    
    #döper den röda exakta 
    t_rod = np.linspace(0, 2*T, 200)
    y_rod = y_exakt(t_rod)
    plt.plot(t_rod, y_rod, 'r', linewidth=2, label="Exakt lösning")
    plt.legend()
    
    #plottens titel
    plt.title(titel)
    
    #axelrubrikerna
    plt.xlabel("t")
    plt.ylabel("y")
    
    #lika avstånd på axklarna
    plt.axis("equal")
    
    #svaga bakgrundsstreck
    plt.grid(True)
    
    #visa plotten
    plt.show()

#Menyfunktion, kalla på funktionerna
def main_F1a():
    plotta_riktningsfalt(t_start_slut, y_start_slut, 
                         t_prim, f, tathet=20, scale=50, 
                         titel = "Riktningsfältet med den exakta lösningen")

main_F1a()






# F1b)
# Approximera metoden med Eulers metod framåt (skriv allmänt så koden kan användas igen) och steglängden h=0.1
# Spara initialdata och alla lösningsvärden i en vektor
# Plotta den numeriska lösningsvektorn som funktion av tiden


def euler_framat(f, a, b, y0, h):

    n = round(np.abs(b-a)/h) # totalaantalet steg
    print(n)
    
    # Diskretiseringsteg: Generera n+1 friddpunkter
    t = np.linspace(a, b, n+1) #anger punkterna i t led i en lista
    y = np.zeros(n+1) #alla y värden ska vara [0,0,0,0,0]

    # Begynnelsevillkor
    y[0] = y0

    # Iterera med Eulers framåt. Formler EF.1
    # Modul 6 Del 2
    for k in range(n):
        y[k+1] = y[k] + h*f(t[k], y[k])
    return t, y


def skapa_tabell(tsol, ysol, h):

    yexakt = y_exakt(tsol)
    Ek_global = np.abs(ysol-yexakt)
    print("felet ek")
    print(Ek_global)

    tabell = {"Steg: tk": tsol,
              "yk": ysol
              }
    df = pd.DataFrame(tabell)
    print(df)


# F1c)
# Verifiera att felet blir Ek = |yk(T)−yexakt(T)| ≈ 0.0188

def main_F1b():
    a = 0 #starttid
    b = 1.2 #sluttid
    y0 = 1 #värdet vid starttiden
    h = 0.1 #steglängd
    # a)
    ta, ya = euler_framat(f, a, b, y0, h)
    print("ya")
    print(ya[-1])
    print("ta")
    print(ta[-1])
    yexakt = y_exakt(ta[-1]) # ta = 1.2
    Ek = np.abs(ya[-1]-yexakt)
    print(Ek)
    # plotta_solution(ta,ya,h)
    skapa_tabell(ta, ya, h)

    # f = lambda t, y: -y      #Definera en lambda funktion (=inline function)



main_F1b()


# -----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga F2"""
# F2a, b och c)
# Gör en konvergensstudie genom att ha h och beräkna numerisk lösning med Eulers metod framåt
# Spara lösningen för varje h i y(T)

def main_f2():
    fel_lista = []
    h_lista = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    a = 0
    b = 1.2
    y0 = 1
    h = 0.1
    # Beräkna felen ek = |yk(T) − yexakt(T)| och verifiera att felen blir enligt facit
    for h in h_lista:
        ta, ya = euler_framat(f, a, b, y0, h)
        yexakt = y_exakt(ta[-1])
        Ek = np.abs(ya[-1]-yexakt)
        fel_lista.append(Ek)
        print("för h = ", h, "så är lösningen", ya[-1], "och felet Ek = ", Ek)

    print(" \nuppgift f2c, nogranhetsordningarna är:")

    p_lista = []

    # Beräkna nogrannhetsordningen empiriskt och jämför
    for i in range(len(fel_lista) - 1):
        ek_h = fel_lista[i]
        ek_h_halverad = fel_lista[i+1]
        p = np.log(ek_h / ek_h_halverad) / np.log(2)
        p_lista.append(p)
        print(p)
    # print(p_lista)

main_f2()

# -----------------------------------------------------------------------------------------------------------------
"""Följande är svar på fråga T1"""
# T1a)
# Gör en differentialekvation till en LCR krets genom att skriva om ekvation 2 till y´= F(t,y)

# T1b)

#Skriv differentialekvationen
def diffekvation_2(t, y, R, L, C):
    # t, y, R, L, C är inparametrar. Funktionen returnerar vektorn F(t, y).
    q = y[0]  # detta är för solve_ivp
    i = y[1]  # första derivatan

    q_prim = i
    i_prim = (-(R/L)*i)-((1/(L*C))*q)

    return np.array([q_prim, i_prim])


# 2. Definiera tidsintervall och initialvillkor
t_span = (0, 5)
y0 = [1]


def lös_ODE(R, L, C, t_span, Q0):
    #(resistansen, L induktans, C kapacitans, tidspannet, initial laddning)
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
"""Följande är svar på fråga T2"""
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
    A = np.round(mittersta_diagnonalen + nedre_diagnonalen + ovre_diagnonalen) 
    #A har 3 diagonaler, i formen av (N-1)*(N-1)
    
    #Nu ska x_vektorn skrivas
    x_j = np.linspace(0+h, L-h, N-1) 
    #x_j värden skrivs i en lista, där indexet j = [1, 2, 3, …, (N-1))]
    #x_j elementen står för punkterna där vi ska approximera temperaturen
    #np.linspace(startpunkt, slutpunkt, antal x_j punkter som vi ska räkna ut)
    #https://numpy.org/devdocs//reference/generated/numpy.linspace.html 
    
    #Nu ska b_vektorn skrivas
    b = q(x_j).astype(float)
    #Högerledsvektorn b har längden (N-1)
    b[0] -= (k / h**2) * TL 
    b[-1] -= (k / h**2) * TR
    
    #Lös ut T inre och 
    T_ej_med_randvillkor = np.linalg.solve(A,b)

    x = np.linspace(0, L, N+1) #alla x_värden, dvs randvärderna är inkluderade
    T = np.concatenate(([TL],T_ej_med_randvillkor,[TR])) #alla T_värden, dvs randvärderna är inkluderade
    
    return A, x_j, x, b, T

#printa temperaturerna i en lista
A, x_j, x, b, T = stavens_temperatur(N, q, k, TL, TR, L)
print("A:\n", A)
print("x värden:\n", x_j)
print("b värden:\n", b)
print("Stavens temperatur beräknat på",N+1,"stycken punkter är: \n",T)

# T2d) KLAR
#Få fram A, x och b för N=100 istället.
A_T2d, x_j_T2d, x_T2d, b_T2d, T_T2d = stavens_temperatur(100, q, k, TL, TR, L)
#A_T2d systemmatrisen 99*99 för de inre punkterna
#x_T2d alla x koordinaterna där randvärderna är inkluderade
#b_T2d b för de inre punkterna
#T_T2d temperaturen där randvärderna är inkluderade

#Skriv funktionen som plottar temperaturfördelningen
def plotta_stavens_temperatur(x,T): 
    #x = vad som ska vara på x-axeln
    #T = vad som ska vara på y-axeln (som ska vara temperaturen)
    plt.plot(x, T, '-o', markersize=2.5) #markersize ändrar punkternas storlek
    plt.title("Stavens temperaturfördelning (N = 100)")
    plt.xlabel("x")
    plt.ylabel("T(x), temperaturen")
    plt.grid(True) #Visar hjälplinjerna

#kalla på funktionen som plottar
plotta_stavens_temperatur(x_T2d,T_T2d)

#Printa vad temperaturen uppskattas vara i x=0.2
for idx, element in enumerate(x_T2d):
    #https://stackoverflow.com/questions/522563/how-can-i-access-the-index-value-in-a-for-loop
    if element == 0.2:
        print("Stavens temperatur i x = 0.2 är", T_T2d[idx])


# T2e) KLAR
#Antal delintervall i lista, börja på N=50
N_T2e = [50, 100, 200, 400, 800, 1600]

#Jämförelsevärde T(0.7) = 1.6379544 enligt uppgift
T_jamforelsevarde_noll_komma_sju_avrundat= 1.6379544

# Gör konvergensstudie med steglängdshalvering.
felen = [] #tom lista som fyller på för att beräkna p

#Gör kolumnrubriken
print(" \n  N     h        T(0.7)     |felet|")

#Lägg till värden i felen
for N in N_T2e: #för N i listan N_T2e

    #Skapa en matris A och b, alla x värden och T värden för varje element i N_T2e genom att anropa funktionen stavens_temperatur
    A, x_j, x, b, T = stavens_temperatur(N, q, k, TL, TR, L) 
    
    #ibland finns inte x = 0.7 ooch det löses
    T_noll_komma_sju_approxiamtion = np.interp(0.7, x, T)
    #interpolationden gör en linje mellan approximationspunktern nära x = 0.7 för att x värdet ska bli exakt 0.7
    #https://numpy.org/doc/2.3/reference/generated/numpy.interp.html 
    
    #Skriv absoluta felet
    varje_fel = abs(T_noll_komma_sju_approxiamtion - T_jamforelsevarde_noll_komma_sju_avrundat) 
    
    #appendar varje absoluta fel till listan felen
    felen.append(varje_fel) 
    
    print(f"{N:4}  {L/N:.3e}  {T_noll_komma_sju_approxiamtion:.6}  {varje_fel:.3e}")
    #{N:4} N värden ska ta upp 4 "fält" i bredd. f"{N:4}" gör att det inte behöver stå format(N, "4")
    #https://www.w3schools.com/python/ref_string_format.asp
    #{L/N:.3e} tre decimalers nogrannhet 
    #e = 10^(ett tal som ska anges)
    #{T_noll_komma_sju_approxiamtion:.6} temperaturen för varje värde på N_T2e avrundat till sex decimalers nogrannhet
    #{varje_fel:.3e} tre decimaler där man använder e

#Skriv nogrannhetsordningen
print("\nNogrannhetsordning p ges av:")
for i in range(len(N_T2e)-1): #vi sätter minus 1 för att garantera att det finns ett element som heter felen[i+1]
    if felen[i+1] == 0:
        print("Det går inte att dela med noll")
    else:
        p = np.log(felen[i] / felen[i+1]) / np.log(2)
        print("Vi jämför värden från", N_T2e[i], "och", N_T2e[i+1], "st delintervall som ger p ≈",p)
        #nogrannhetsordningen verkar stämma väl överens med teorin. 


# T2f) KLAR
#Testa att ändra randvillkoren TL och TR och se om det ändras.
#Vi ser att det ändras
print("\nVi ser att randvillkoren ändrar temperaturfördelningen")
#Det är ändå en konvex struktur.

