from scipy.optimize import minimize
import numpy as np
from numpy import random
import time
# функция для расчета значений целевой функции
def func(x):
    return (-x[0]**2-1.5*x[1]**2+2*x[0]*x[1]-4*x[0]+8*x[1])

# функция вычисления градиента целевой функции
def func_deriv(x):
    dfx0 = -2*x[0]+2*x[1]-4
    dfx1 = -3*x[1]+2*x[0]+8
    return np.array([dfx0, dfx1])

# кортеж для задания ограничений - каждое ограничение - словарь с описанием типа условия, функции вычисления левой части и функкции вычисления градиента левой части
cons = ({'type':'ineq', 'fun': lambda x: np.array([-x[0]-x[1]+3]), 'jac': lambda x: np.array([-1,-1])},
{'type':'ineq', 'fun': lambda x: np.array([-x[0]+x[1]+1]), 'jac': lambda x: np.array([-1,1])},
{'type':'ineq', 'fun': lambda x: np.array([x[0]]), 'jac': lambda x: np.array([1,0])},
{'type':'ineq', 'fun': lambda x: np.array([x[1]]), 'jac': lambda x: np.array([0,1])})

#случайное число для начального прогона
def random_cord(x):
    x1 = x[0]
    x2 = x[1]
    while True:
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(1, 3)
        if in_set(([x1, x2])):
            break
    return x1, x2

# функция для проверки условий
def in_set(x):
    return (x[0]+x[1]<=3) and (x[0]-x[1]<=1) and (x[0]>=0) and (x[1]>=0)

#случайное число для отжига, которе подходит под ограничения
def random_x(xold, t):
    a = random.normal(0,1)
    x1 = xold[0] + t*a
    x2 = xold[1] + t*a
    while True:
        if in_set(([x1, x2])):
            break
        a = random.normal(0, 1)
        x1 = xold[0] + t * a
        x2 = xold[1] + t * a
    return x1, x2


# метод отжига
def otg(fbegin, x, t0):
    # начальная инициализация
    fopt = fbegin  # оптимум
    fxi = 0  # значение целевой функции при след итерации
    t = t0  # температура
    xopt = x  # значение параметра, при котором достигается оптимум целевой функции
    iteration = 1  # колличество итерация
    ex = 0  # для условия выхода
    x1, x2 = random_cord(x)  # Случайный выбор текущей точки x(0)
    fxi = func(([x1, x2]))  # и вычисление целевой функции F(x(0)) для данной точки. ЕслиF(x(0))<Fbegin, то Fopt=F(x(0))
    if fxi < fbegin:
        fopt = fxi
        xopt = ([x1, x2])

    while ex == 0:
        x1, x2 = random_x(xopt, t)  # Генерация новой точки x(i).
        fxi = func(([x1, x2]))

        if fxi < fopt:  # если значение целевой функции < текущего оптимума, обновляем оптимум и точки
            xopt = ([x1, x2])
            fopt = fxi

        else:  # иначе считаем вероятность
            a = np.random.uniform(0, 1)  # генерируем случайную величину a, равномерно распределенную на интервале [0,1)
            p = np.exp(-(fxi - fopt)/t)


            if p > a:  # если вероятность больше случайного числа, меняем оптимум и точки
                xopt = ([x1, x2])
                fopt = fxi
                t = t0/np.log(1+iteration)  # уменьшаем температутру

        if iteration == 10000:  # проверка на выход из алгоритма
            ex = 1
        iteration += 1
    return xopt, fopt, iteration, t


#вывод результатов методов
start = time.time()
x = ([0, 3])
x1, x2 = random_cord(x)
fbgn = func(x)
xopt, fopt, iteration, t = otg(fbgn, x, 100)
stop = time.time()
print("Time:", stop - start)
print('Значения приводящие к оптимуму = ', xopt)
print('Оптимум = ', fopt)
print('Итерация= ', iteration)
print('Температура = ', t)
with open('res2.txt', 'a') as outFile:
    outFile.write("Time:" + str(stop - start) + '\n' + 'xopt = ' + str(xopt) + '\n' + 'fun = ' + str(fopt) + '\n' + "------------" +  '\n')
print('__________')
start = time.time()
res = minimize(func,x,jac=func_deriv, constraints=cons)
stop = time.time()
print("Time:", stop - start)
print(res)
print('_________')





