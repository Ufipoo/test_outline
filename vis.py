import numpy as np
import math
import matplotlib.pyplot as plt

def f(x):
    return (2/math.sqrt(math.pi))*math.exp(-x**2)

def f1(x, eps):
    n = 0
    s = 0.0
    a = x
    while(abs(a)>eps):
        s += a
        a *= (-1)*(x*x)*(2*n+1)/((n+1)*(2*n+3))
        n += 1
    s *= 2/math.sqrt(math.pi)
    return s


def lagrang(x, y, t):
    sum = 0.0
    for j in range(0, np.size(x)):
        p = 1.0
        for i in range(0, np.size(x)):
            if i != j:
                p = p*(t-x[i])/(x[j]-x[i])
        sum = sum+y[j]*p
    return sum


def cheb(a, b, k):
    x = np.zeros(k+1)
    for i in range(0, np.size(x)):
        x[i] = 0.5*(a+b) + 0.5*(b-a)*np.cos(math.pi*(2*i + 1)/(2*k+2))
    return x

def create_list(llim, rlim, n):
    step = (rlim - llim) / n
    return list(map(lambda i: llim + step * i, range(n)))

e_ru = []
e_ch = []
e_ru.append(0.0)
e_ch.append(0.0)
kol2 = []
kol2.append(0.0)

for q in range(1, 28):
    k = 11
    e = 0.000001
    a = 0.0
    b = 2.0
    n = q
    h = (b-a)/(n)
    xs = create_list(a, b, q)
    ys = np.zeros(np.size(xs))
    x = np.linspace(a, b, k)  # xs1
    for i in range(0, np.size(xs)):
        ys[i] = f1(xs[i], e)

    es = []
    for i in x:
        es.append(abs(f1(i, e) - lagrang(xs, ys, i)))
    e_ru.append(max(es))
    kol2.append(q)

for q in range(1, 700):
    k = 11
    e = 0.000001
    a = 0.0
    b = 2.0
    n = q
    h = (b - a) / (n)
    xs = cheb(a, b, q)
    ys = np.zeros(np.size(xs))
    x = np.linspace(a, b, k)  # xs1
    for i in range(0, np.size(xs)):
        ys[i] = f1(xs[i], e)

    es = []
    for i in x:
        es.append(abs(f1(i, e) - lagrang(xs, ys, i)))
    e_ch.append(max(es))
    if q not in kol2:
        kol2.append(q)


for i in range(0, 28):
    if i % 6 == 0:
        print(e_ru[i])
print(e_ru[20])
print(e_ru[-1])

plt.clf()
plt.plot(kol2[:28], e_ru)
plt.title('График максимальных погрешностей при разных n(количество узлов).')
plt.xlabel('n')
plt.ylabel('максимальная погрешность')
plt.show()
print('________')
for i in range(0, 28):
    if i % 6 == 0:
        print(e_ch[i])
print(e_ch[20])
print(e_ch[-1])
plt.clf()
plt.plot(kol2, e_ch)
plt.title('График максимальных погрешностей при разных n(количество узлов).')
plt.xlabel('n')
plt.ylabel('максимальная погрешность')
plt.show()


k = 11
e = 0.000001
a = 0.0
b = 2.0


fx = np.zeros(k)
x = np.linspace(a, b, k)

for i in range(0, np.size(fx)):
    fx[i] = f1(x[i], e)
print('fx: ')
print(fx)


def L_Rec(X, n):
    h = X / n
    ans=0
    for i in range(0, n):
        x = h * i
        ans += f(x) * h
    return round(ans, 9), n, h

list_x = []
list_n = []
L_R=[]
for i in range(0, len(x)):
    num=2
    a1=L_Rec(x[i], num)
    a2=L_Rec(x[i], num*2)
    while math.fabs(a1[0]-a2[0]) > e:
        num *= 2
        a1=a2
        a2=L_Rec(x[i], num*2)
        if x[i] == 1.0:
            list_x.append(abs(fx[i] - a1[0])/a1[2])
            list_n.append(a1[1])

    L_R.append(a1[0])
    print(a1, round(abs(fx[i] - a1[0]), 9))
    if x[i] == 1.0:
        print(list_x)
        print(list_n)


print('______________________________________')

def C_Rec(X, n):
    h = X / n
    ans=0
    for i in range(0,n):
        a = h * i
        b = a + h
        ans += f((a+b)/2)
    ans *= h
    return round(ans,9), n, h

list_x = []
list_n = []
C_R=[]
for i in range(0, len(x)):
    num=2
    a1=C_Rec(x[i], num)
    a2=C_Rec(x[i], num*2)
    while(abs(a1[0]-a2[0])>e):
        num *= 2
        a1=a2
        a2=C_Rec(x[i], num * 2)
        if x[i] == 1.0:
            list_x.append(abs(fx[i] - a1[0]) / a1[2]**2)
            list_n.append(a1[1])
    C_R.append(a1[0])
    print(a1, round(abs(fx[i] - a1[0]), 9))
    if x[i] == 1.0:
        print(list_x)
        print(list_n)



print('______________________________________')
def Trapez(X, n):
    h= X / n
    ans=0
    for i in range(n):
        a = h * i
        b = a + h
        ans += f(a)+f(b)
    ans *= h/2
    return round(ans,9), n, h

list_x = []
list_n = []
Tr=[]
for i in range(0, len(x)):
    num=2
    a1=Trapez(x[i], num)
    a2=Trapez(x[i], num*2)
    while(abs(a1[0]-a2[0])>e):
        num *= 2
        a1=a2
        a2=Trapez(x[i], num*2)
        if x[i] == 1.0:
            list_x.append(abs(fx[i] - a1[0]) / a1[2]**2)
            list_n.append(a1[1])
    Tr.append(a1[0])
    print(a1, round(abs(fx[i] - a1[0]), 9))
    if x[i] == 1.0:
        print(list_x)
        print(list_n)


print('______________________________________')
def Simpson(X, n):
    h= X / n
    ans=0
    for i in range(n):
        a = h * i
        b = a + h
        ans += (f(a)+4*f((a+b)/2) + f(b))
    ans *= h/6
    return round(ans,9), n, h

list_x = []
list_n = []
Simp=[]
for i in range(0, len(x)):
    num=2
    a1=Simpson(x[i], num)
    a2=Simpson(x[i], num*2)
    while(abs(a1[0]-a2[0])>e):
        num *= 2
        a1=a2
        a2=Simpson(x[i], num*2)
        if x[i] == 1.0:
            list_x.append(abs(fx[i] - a1[0]) / a1[2]**4)
            list_n.append(a1[1])
    Simp.append(a1[0])
    print(a1, round(abs(fx[i] - a1[0]), 9))
    if x[i] == 1.0:
        print(list_x)
        print(list_n)


print('______________________________________')
def Gauss(X, n):
    h= X / n
    ans=0
    for i in range(n):
        a = h * i
        b = a + h
        ans += f(a + h /2 * (1 - 1/np.sqrt(3)))
        ans += f(a + h /2 * (1 + 1/np.sqrt(3)))
    ans *= h/2
    return round(ans,9), n, h

list_x = []
list_n = []
Gau=[]
for i in range(0, len(x)):
    num=2
    a1=Gauss(x[i], num)
    a2=Gauss(x[i], num*2)
    while(abs(a1[0]-a2[0])>e):
        num *= 2
        a1=a2
        a2=Gauss(x[i], num*2)
        if x[i] == 1.0:
            list_x.append(abs(fx[i] - a1[0]) / a1[2]**3)
            list_n.append(a1[1])
    Gau.append(a1[0])
    print(a1, round(abs(fx[i] - a1[0]), 9))
    if x[i] == 1.0:
        print(list_x)
        print(list_n)



plt.plot(x,fx,'y')
plt.plot(x,L_R, "ro")
plt.plot(x,C_R,"r")
plt.plot(x,Tr)
plt.plot(x,Simp)
plt.plot(x,Gau, "bo")
plt.show()