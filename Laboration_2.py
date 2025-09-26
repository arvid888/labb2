#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 19:31:38 2025

@author: mollyfreeman
"""



"""Följande är svar på fråga F1"""
#F1a) 
#Vi ska rita riktiningsfältet för DE och lägg till den exakta lösningen till ekvationen


#F1b) 
#Approximera metoden med Eulers metod framåt (skriv allmänt så koden kan användas igen) och steglängden h=0.1
#Spara initialdata och alla lösningsvärden i en vektor
#Plotta den numeriska lösningsvektorn som funktion av tiden

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
def omskrivning_av_diffekvation(t,y):
    dU/dt = F(t,U)
    U(t0)=U0
    
def euler_framat(t,y): #föreläsningsanteckningarna modul 7 (del 1 av 4) räkneexempel 7.2

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