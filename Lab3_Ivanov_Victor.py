import sys
import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
from prettytable import PrettyTable
from scipy.optimize import fsolve
from sympy import *

########################################################################################################################
'''
                        V                     VH               HV                     V
                         V                   V H               H V                   V
                          V                 V  H               H  V                 V
                           V               V   H               H   V               V
                            V             V    H               H    V             V
                             V           V     H               H     V           V
                              V         V      HHHHHHHHHHHHHHHHH      V         V
                               V       V       H               H       V       V
                                V     V        H               H        V     V
                                 V   V         H               H         V   V
                                  V V          H               H          V V
                                   V           H               H           V
'''
'''====================================================================================================================
                            Лабороторна робота №3.Чисельне розв’язання нелінійних рівнянь та систем
Мета:  закріплення знань із застосування методів чисельного розв’язання 
нелінійних алгебраїчних рівнянь та систем.
Роботу виконував: Іванов Віктор Віталійович, ФФ-03
Роботу перевіряла: Гордійко Н.О.
Варіант №14
Завдання:
а) Визначити один з коренів рівняння x * sin(x) - 1 = 0 на інтервалі [0;3] методом Ньютона з точністю 1e-4. 
Перевірити отриманий корінь та вивести кількість ітерацій;
б) Розв’язати систему рівнянь
    { tg(xy) = x**2
    { 0.7x**2 + 2y**2 = 1
    (x > 0, y > 0)
   з точністю 10e-5 методом Ньютона.Початкове наближення визначити графічно.Вивести к-сть ітерацій;
в) Розв'язати систему б) методом простої ітерації з тією ж точністю. Вивести 
   кількість ітерацій;   
г) Розв’язати систему рівнянь
    { x**2 + y**2 + z**2 - 3 = 0
    { 2*x**2 + y**2 - 4*z - 1 = 0
    { 3*x**2 - 4*y + z**2 = 0
   з початковим наближення (1;1;1) та точністю 1e-3
===================================================================================================================='''

'---------------------------------------------------------------------------------------------------------------------'
"Програмний код"
plt.style.use("cyberpunk")
print("Реалізація методу Ньютона a)")


def Newton(f, dfdx, x, eps):
    xstore = []
    fstore = []
    f_value = f(x)
    iteration_counter = 0
    while abs(f_value) > eps and iteration_counter < 100:
        try:
            x = x - float(f_value) / dfdx(x)
        except ZeroDivisionError:
            print("Error! The derivative is zero for x = ", x)
            sys.exit(1)  # Abort with error
        f_value = f(x)
        xstore.append(x)
        fstore.append(f_value)
        iteration_counter += 1

    if abs(f_value) > eps:
        iteration_counter = -1

    return x, iteration_counter, xstore, fstore


def f(x):
    return x * np.sin(x) - 1


def dfdx(x):
    return np.sin(x) + x * np.cos(x)


solution, no_iterations, xvalues, fvalues = Newton(f, dfdx, x=3, eps=1.0e-4)

if no_iterations > 0:  # Solution found
    print("Number of function calls: %d" % (no_iterations))
    print("Solution: x = %.4f" % (solution))
else:
    print("No solution found!")
print("###############################################################################################################")
print("Чисельне рішення")
func = lambda x1: x1 * np.sin(x1) - 1
x_init = 3.0
x_solution = fsolve(func, x_init)
print("x =", *x_solution.round(4))
print("###############################################################################################################")
########################################################################################################################
def newtonRaphson(x):
    h = f(x) / dfdx(x)
    i = 0
    while abs(h) >= 1.0e-4:
        h = f(x) / dfdx(x)
        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
        i += 1
    i -= 1
    print(f'The value of the root is : {i=}',
          "%.4f" % x)


# Driver program to test above
x0 = 3  # Initial values assumed
newtonRaphson(x0)
########################################################################################################################


def Task1_graphic():
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-20 * np.pi, 20 * np.pi, 100)
    x1 = np.array([i for i in xvalues])
    f1 = np.array(fvalues)
    plt.ylabel(r'$y=f(x)$', fontsize=13, fontname='Arial', color='white')
    plt.xlabel(r'$x$', fontsize=13, fontname='Arial', color='white')
    plt.grid(which='both', linewidth=1.5, linestyle='-', color='gray')
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=8, width=2)
    ax.minorticks_on()
    ax.grid(which='major',
            linewidth=2)
    ax.grid(which='minor',
            linestyle=':')
    ax.plot(x, f(x))
    plt.scatter(x1, f1, color="blue", marker="D", edgecolors='red', linewidth=3, s=10, hatch='||||')
    legend = plt.legend(["xsin(x)-1=0", "Approximate point"], loc='upper right', shadow=True, fontsize='x-large',
                        frameon=True, title="Legend", title_fontsize=15, framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('black')
    frame.set_edgecolor('red')
    plt.title("x*sin(x)-1=0")
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
    mplcyberpunk.add_glow_effects()
    plt.show()
    fig.savefig('xsinx-1_0.png', dpi=300, bbox_inches='tight')


Task1_graphic()


def Newton_graphic():
    x = np.array([i for i in xvalues])
    f = np.array(fvalues)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel(r'$y=f(x)$', fontsize=13, fontname='Arial', color='white')
    plt.xlabel(r'$x$', fontsize=13, fontname='Arial', color='white')
    plt.grid(which='both', linewidth=1.5, linestyle='-', color='gray')
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=8, width=2)
    ax.minorticks_on()
    ax.grid(which='major',
            linewidth=2)
    ax.grid(which='minor',
            linestyle=':')
    plt.scatter(x, f, color="blue", marker="D", edgecolors='red', linewidth=3, s=60, hatch='||||')
    plt.plot(xvalues, fvalues, color="yellow")
    legend = plt.legend(["Method Newton's"], loc='upper right', shadow=True, fontsize='x-large',
                        frameon=True, title="Legend", title_fontsize=15, framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('black')
    frame.set_edgecolor('red')
    plt.title("Convergence diagram")
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
    mplcyberpunk.add_glow_effects()
    plt.show()
    fig.savefig('Convergence diagram_Task1.png', dpi=300, bbox_inches='tight')


Newton_graphic()
########################################################################################################################
print("Метод Ньютона б)\n")
print("First realise code")


def equations(p):
    x, y = p
    return (np.tan(x * y) - x * x, 0.7 * x * x + 2 * y * y - 1)


x, y = fsolve(equations, (0.5, 0.5), xtol=1e-5)
print(f'Answers: {x=}, {y=}')
print(f'Check answers, precision:', equations((x, y)), '\n')


def function_exercise(xy):
    x, y = xy
    return [np.tan(x * y) - x * x,
            0.7 * x * x + 2 * y * y - 1]


def jacobian_exercise(xy):
    x, y = xy
    return [[(y / (np.cos(x * y) ** 2)) - 2 * x, (x / (np.cos(x * y) ** 2))],
            [1.4 * x, 4 * y]]


def iter_newton(X, function, jacobian, imax=1e6, tol=1e-5):
    table = PrettyTable()
    table.field_names = ["i", "X", "Y"]
    for i in range(int(imax)):
        J = jacobian(X)  # calculate jacobian J = df(X)/dY(X)
        Y = function(X)  # calculate function Y = f(X)
        dX = np.linalg.solve(J, Y)  # solve for increment from JdX = Y
        X -= dX  # step X by dX
        i += 1
        table.add_row([i, X, Y])
        # print(f'Number of iterations: {i=}\t Result: {X,Y=}')
        if np.linalg.norm(dX) < tol:  # break if converged
            print('Converged')
            break
    print(table)
    return X


X_0 = np.array([0.5, 0.5], dtype=float)
print(iter_newton(X_0, function_exercise, jacobian_exercise))
########################################################################################################################


print("\nSecond realise code")


def jacobian_exercise1(x, y):
    return [[(y / (np.cos(x * y) ** 2)) - 2 * x, (x / (np.cos(x * y) ** 2))],
            [1.4 * x, 4 * y]]


jotinha = (jacobian_exercise1(0.5, 0.5))


def negative(lst):
    return [-i for i in lst]


def function_exercise1(x, y):
    return np.tan(x * y) - x * x, 0.7 * x * x + 2 * y * y - 1


bezao = (function_exercise1(0.5, 0.5))


def x_delta_by_gauss(J, b):
    return np.linalg.solve(J, b)


x_delta_test = x_delta_by_gauss(jotinha, bezao)


def x_plus_1(x_delta, x_previous):
    x_next = x_previous + x_delta
    return x_next


print("Errors: ", x_plus_1(x_delta_test, [0.5, 0.5]))


def newton_method(x_init):
    first = x_init[0]
    second = x_init[1]
    jacobian = jacobian_exercise1(first, second)
    vector_b_f_output = negative(function_exercise1(first, second))
    x_delta = x_delta_by_gauss(jacobian, vector_b_f_output)
    x_plus_1 = x_delta + x_init
    return x_plus_1


def iterative_newton(x_init):
    counter = 0
    x_old = x_init
    x_new = newton_method(x_old)
    diff = np.linalg.norm(x_old - x_new)
    while diff > 1e-5:
        counter += 1

        x_new = newton_method(x_old)

        diff = np.linalg.norm(x_old - x_new)

        x_old = x_new

    convergent_val = x_new
    print("Number of iterations: ", counter)
    return convergent_val


print("Result: ", iterative_newton([0.5, 0.5]), "\n")


def Newton_sys_graphic():
    fig1, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-np.pi, np.pi, 100)
    y = np.linspace(-np.pi, np.pi, 100)
    p, q = np.meshgrid(x, y)

    f1 = lambda x, y: np.tan(x * y) - x**2
    f2 = lambda x, y: 0.7*x**2 + 2*y**2
    z = -f1(p, q)
    w = f2(p, q)

    # рисуем линии уровня f1(x,y)==0 и f2(x,y)==1
    plt.contour(p, q, z, [0], colors=["k"])
    plt.contour(p, q, w, [1], colors=["r"])
    plt.scatter(*function_exercise1(0.5, 0.5), color="blue", marker="d", edgecolors='yellow',
                linewidth=8, s=30, hatch='+o')
    plt.ylabel(r'$y$', fontsize=13, fontname='Arial', color='white')
    plt.xlabel(r'$x$', fontsize=13, fontname='Arial', color='white')
    plt.grid(which='both', linewidth=1.5, linestyle='-', color='gray')
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=8, width=2)
    ax.minorticks_on()
    ax.grid(which='major',
            linewidth=2)
    ax.grid(which='minor',
            linestyle=':')
    plt.title("Function Graphs")
    plt.show()
    fig1.savefig('newton_point.png', dpi=300, bbox_inches='tight')


Newton_sys_graphic()
########################################################################################################################
print("Система рівнянь г)")


def eq2(p):
    x, y, z = p
    return x**2 + y**2 + z**2 - 3, 2*x**2 + y**2 - 4*z - 1, 3*x**2 - 4*y + z**2


x, y, z = fsolve(eq2, (1, 1, 1), xtol=1e-3)
print('Відповідь: ', x, y, z)
print('Перевірка рішення: ', eq2((x, y, z)), "\n")

########################################################################################################################
def different():
    x, y = symbols('x y')
    F1 = atan(x*x) / y
    F2 = sqrt((1-0.7*x*x) / 2)
    fi1x = diff(F1, x)
    fi1y = diff(F1, y)
    fi2x = diff(F2, x)
    fi2y = diff(F2, y)
    print(f'{fi1x=} \t {fi1y=} \n {fi2x=} \t {fi2y}')


different()
########################################################################################################################

########################################################################################################################
print('Метод простої ітерації та метод Зейделя')


def f(x, y):
    return np.tan(x * y) ** 0.5, ((1 - 0.7 * x * x) / 2) ** 0.5


mytable = PrettyTable()
mytable.field_names = ["Iterations", "x", "y"]
x, y = 0.5, 0.5
eps = 1e-5
x_old = y_old = np.Inf
counter = 0
while max(abs(x_old - x), abs(y_old - y)) > eps:
    x_old, y_old = x, y
    x, y = f(x_old, y_old)
    if max(x, y) > 1 or counter > 1000:
        print("Didn't fit")
        break
    counter += 1
    mytable.add_row([counter, x, y])

print(mytable)

vtablev = PrettyTable()
vtablev.field_names = ["i", "x", "y"]
x, y = 0.5, 0.5
for i in range(10):
    i += 1
    x = tan(sqrt(x*y))
    y = .5*(1-.7*x)
    # print(sqrt(x), sqrt(y))
    vtablev.add_row([i, sqrt(x), sqrt(y)])

print(vtablev)
########################################################################################################################
'''Висновок: в результаті цієї лабороторної роботи були обработані разні обчислювальні методи.
Закріпили знання із застосуванням методів чисельного розв’язання 
нелінійних алгебраїчних рівнянь та систем.'''
