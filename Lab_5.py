import random
import numpy as np
import copy
import math
import scipy.stats

x_min = [-6, -8, -10]
x_max = [10, 3, 9]


x_0 = [(x_min[i] + x_max[i])/2 for i in range(len(x_min))]
x_delta = [x_max[i] - x_0[i] for i in range(len(x_max))]

x_average_min = sum(x_min) / 3
x_average_max = sum(x_max) / 3

y_min = 200 + int(x_average_min)
y_max = 200 + int(x_average_max)

M = 3
N_lineal = 4
N_interaction = 8
N_squares = 15
l = 1.215

list_coefficients = []

cohren_values = {1: 9065, 2: 7679, 3: 6841, 4: 6287, 5: 5892, 6: 5598, 7: 5365, 8: 5175, 9: 5017,
                 10: 4884, range(11, 17): 4366, range(17, 37): 3720, range(37, 145): 3093}


def det(arr):
    return np.linalg.det(np.array(arr))


def y_lineal_regression(k0=0, k1=0, k2=0, k3=0, x1=0, x2=0, x3=0):
    return k0 + k1 * x1 + k2 * x2 + k3 * x3


def y_interaction_regression(b0=0, b1=0, b2=0, b3=0, b12=0, b13=0, b23=0, b123=0, x1=0,
                             x2=0, x3=0, x1x2=0, x1x3=0, x2x3=0, x1x2x3=0):
    return b0 + b1 * x1 + b2 * x2 + b3 * x3 + b12 * x1x2 + b13 * x1x3 + b23 * x2x3 + b123 * x1x2x3


def y_squares_regression(b0=0, b1=0, b2=0, b3=0, b12=0, b13=0, b23=0, b123=0, b11=0, b22=0, b33=0, x1=0,
                         x2=0, x3=0, x1x2=0, x1x3=0, x2x3=0, x1x2x3=0, x1x1=0, x2x2=0, x3x3=0):
    return b0 + b1 * x1 + b2 * x2 + b3 * x3 + b12 * x1x2 + b13 * x1x3 + b23 * x2x3 + b123 * x1x2x3 + b11 * x1x1 +\
           b22 * x2x2 + b33 * x3x3


def get_cohren_criterion(f1):
    for key in cohren_values.keys():
        if type(key) == int:
            if f1 == key:
                value = cohren_values.get(key)
                break
        else:
            if f1 in key:
                value = cohren_values.get(key)
                break
    else:
        value = 2500
    return value / 10000


matrix_x_code_1 = [[-1, -1, -1],
                   [-1, +1, +1],
                   [+1, -1, +1],
                   [+1, +1, -1]]


matrix_x_code_2 = [[-1, -1, -1],
                   [-1, +1, +1],
                   [+1, -1, +1],
                   [+1, +1, -1],
                   [-1, -1, +1],
                   [-1, +1, -1],
                   [+1, -1, -1],
                   [+1, +1, +1]]

matrix_x_code_3 = [[-1, -1, -1],
                   [-1, +1, +1],
                   [+1, -1, +1],
                   [+1, +1, -1],
                   [-1, -1, +1],
                   [-1, +1, -1],
                   [+1, -1, -1],
                   [+1, +1, +1],
                   [-l, 0, 0],
                   [l, 0, 0],
                   [0, -l, 0],
                   [0, l, 0],
                   [0, 0, -l],
                   [0, 0, l],
                   [0, 0, 0]]


class Calc:
    def __init__(self, main_matrix, m, n):
        self.N = n
        self.matrix_x_code, self.matrix_x_natural, self.matrix_x_norm, self.matrix_y = main_matrix, [], [], []
        self.matrix_x_norm_full = []
        self.average_y, self.m_x, self.list_a_i, self.list_a_ii = None, None, None, None
        self.amount_of_x, self.my, self.a12, self.a23, self.a31 = None, None, None, None, None
        self.denominator, self.change, self.list_b, self.list_a = None, None, None, None
        self.list_delta, self.list_x0 = [], []
        self.m, self.p = m, 0.95
        self.q = 1 - self.p
        self.dispersion, self.Coh_coefficient, self.Coh_criterion = None, None, None
        self.f1, self.f2, self.f3, self.f4 = None, None, None, None
        self.d_reproducibility, self.d_evaluation = None, None
        self.list_beta, self.list_t, self.Stud_criterion = [], [], None
        self.d, self.d_adequacy, self.Fish_coefficient, self.Fish_criterion = None, None, None, None
        self.check_adequacy = None
        self.list_y = None
        self.list_coefficients = None
        if self.N == 4:
            self.main_caption = ["X1", "X2", "X3"]
        elif self.N == 8:
            self.main_caption = ["X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3"]
            self.list_b_natural, self.list_b_norm = None, None
            self.string_b_natural, self.string_b_norm = None, None
            self.list_of_indexes = None
        else:
            self.main_caption = ["X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "X1X1", "X2X2", "X3X3"]
            self.list_b_natural = None
            self.string_b_natural = None
            self.list_of_indexes = None

        self.average_y_caption, self.dispersion_caption = [], []
        for i in range(self.m):
            self.main_caption.append("Y" + str(i + 1))
        for i in range(self.N):
            self.average_y_caption.append("Y" + str(i + 1))
            self.dispersion_caption.append("D" + str(i + 1))
        self.string_main1, self.string_main2, self.string_y_average1, self.string_y_average2 = None, None, None, None
        self.string_b, self.string_a, self.matrix_out = None, None, None
        self.string_regression_caption, self.string_regression = None, None

        self.string_dispersion1, self.string_dispersion2 = None, None
        self.string_cohren, self.string_cohren_result1, self.string_cohren_result2 = None, None, None

        self.string_student_marks, self.string_t = None, None
        self.string_student_format, self.string_regression_caption_st = None, None
        self.string_student_regression = None

        self.string_fisher_d, self.string_fisher = None, None
        self.string_fisher_result1, self.string_fisher_result2, self.string_fisher_result3 = None, None, None
        self.values_regression_st = []
        if self.N == 4:
            self.initial_calculation()

    def initial_calculation(self, matrix_y=None):
        self.amount_of_x = len(self.matrix_x_code[0])
        if self.N == 4:
            self.matrix_y = [[random.randint(y_min, y_max + 1) for j in range(self.m)] for i in range(self.N)]
            self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(self.N)]

            for i in range(self.amount_of_x):
                self.list_delta.append(abs(x_max[i] - x_min[i]) / 2)
                self.list_x0.append((x_max[i] + x_min[i]) / 2)

            for i in range(self.N):
                self.matrix_x_natural.append([])
                self.matrix_x_norm.append([])
                for j in range(self.amount_of_x):
                    self.matrix_x_natural[i].append(x_min[j]) if self.matrix_x_code[i][j] == -1 \
                        else self.matrix_x_natural[i].append(x_max[j])
                    self.matrix_x_norm[i].append((self.matrix_x_natural[i][j] - self.list_x0[j]) /(self.list_delta[j]))

            self.matrix_x_norm_full = copy.deepcopy(self.matrix_x_norm)
            for i in range(self.N):
                self.matrix_x_norm_full[i].insert(0, 1)
        elif self.N == 8:
            if matrix_y:
                self.matrix_y = matrix_y[:]
                for i in range(4, self.N):
                    self.matrix_y.append([])
                    for j in range(self.m):
                        self.matrix_y[i].append(random.randint(y_min, y_max + 1))
            else:
                for i in range(self.N):
                    self.matrix_y.append([])
                    for j in range(self.m):
                        self.matrix_y[i].append(random.randint(y_min, y_max + 1))
            self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(self.N)]

            self.matrix_x_natural = []
            for i in range(self.N):
                self.matrix_x_natural.append([])
                for j in range(self.amount_of_x):
                    self.matrix_x_natural[i].append(x_max[j]) if self.matrix_x_code[i][j] == 1 \
                        else self.matrix_x_natural[i].append(x_min[j])

            self.matrix_x_norm = copy.deepcopy(self.matrix_x_code)
            for row in range(self.N):
                self.matrix_x_natural[row].append(self.matrix_x_natural[row][0] * self.matrix_x_natural[row][1])
                self.matrix_x_natural[row].append(self.matrix_x_natural[row][0] * self.matrix_x_natural[row][2])
                self.matrix_x_natural[row].append(self.matrix_x_natural[row][1] * self.matrix_x_natural[row][2])
                self.matrix_x_natural[row].append(
                    self.matrix_x_natural[row][0] * self.matrix_x_natural[row][1] * self.matrix_x_natural[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][1])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][1] * self.matrix_x_norm[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][1] *
                                               self.matrix_x_norm[row][2])
            self.amount_of_x = len(self.matrix_x_norm[0])

            self.matrix_x_norm_full = copy.deepcopy(self.matrix_x_norm)
            for i in range(self.N):
                self.matrix_x_norm_full[i].insert(0, 1)
        else:
            if matrix_y:
                self.matrix_y = matrix_y[:]
                for i in range(8, self.N):
                    self.matrix_y.append([])
                    for j in range(self.m):
                        self.matrix_y[i].append(random.randint(y_min, y_max + 1))
            else:
                for i in range(self.N):
                    self.matrix_y.append([])
                    for j in range(self.m):
                        self.matrix_y[i].append(random.randint(y_min, y_max + 1))
            self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(self.N)]

            self.matrix_x_natural = []
            for i in range(self.N):
                self.matrix_x_natural.append([])
                for j in range(self.amount_of_x):
                    norm_value = self.matrix_x_code[i][j]
                    if norm_value == 1:
                        self.matrix_x_natural[i].append(x_max[j])
                    elif norm_value == -1:
                        self.matrix_x_natural[i].append(x_min[j])
                    elif norm_value == -l:
                        self.matrix_x_natural[i].append(round(-l * x_delta[j] + x_0[j], 2))
                    elif norm_value == l:
                        self.matrix_x_natural[i].append(round(l * x_delta[j] + x_0[j], 2))
                    elif norm_value == 0:
                        self.matrix_x_natural[i].append(x_0[j])

            self.matrix_x_norm = copy.deepcopy(self.matrix_x_code)
            for row in range(self.N):
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][0] *
                                                        self.matrix_x_natural[row][1], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][0] *
                                                        self.matrix_x_natural[row][2], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][1] *
                                                        self.matrix_x_natural[row][2], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][0] * self.matrix_x_natural[row][1] *
                                                  self.matrix_x_natural[row][2], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][0] *
                                                        self.matrix_x_natural[row][0], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][1] *
                                                        self.matrix_x_natural[row][1], 2))
                self.matrix_x_natural[row].append(round(self.matrix_x_natural[row][2] *
                                                        self.matrix_x_natural[row][2], 2))

                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][1])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][1] * self.matrix_x_norm[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][1] *
                                               self.matrix_x_norm[row][2])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][0] * self.matrix_x_norm[row][0])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][1] * self.matrix_x_norm[row][1])
                self.matrix_x_norm[row].append(self.matrix_x_norm[row][2] * self.matrix_x_norm[row][2])
            self.amount_of_x = len(self.matrix_x_norm[0])

            self.matrix_x_norm_full = copy.deepcopy(self.matrix_x_norm)
            for i in range(self.N):
                self.matrix_x_norm_full[i].insert(0, 1)
        self.initial_format()
        self.find_coefficients()

    def initial_format(self):
        if self.N == 4:
            self.string_main1 = "№ " + "{:^6}   " * self.amount_of_x + "{:^6}  " *(
                    len(self.main_caption) - self.amount_of_x)
            self.string_main2 = "{} " + "| {:^4} | " * self.amount_of_x + "|{:^5}| " *(
                    len(self.main_caption) - self.amount_of_x)
        elif self.N == 8:
            self.string_main1 = "№ " + "{:^6}   " * 3 + "{:^8}   "*3 + "{:^10}  " + "{:^8}  " *(
                    len(self.main_caption) - self.amount_of_x)
            self.string_main2 = "{} " + "| {:^4} | " * 3 + "| {:^6} | " * 3 + "| {:^7} | " + "|{:^7}| " *(
                    len(self.main_caption) - self.amount_of_x)
        else:
            self.string_main1 = "№ " + "{:^10} " * 3 + "{:^12} " * 3 + "{:^13}"+ "{:^12} " * 3 + "{:^12}" *(
                    len(self.main_caption) - self.amount_of_x)
            self.string_main2 = "{} " + "| {:^6} | " * 3 + "| {:^8} | " * 3 + "| {:^8} | " + "| {:^8} | " * 3 \
                                + "|{:^9}| " *(len(self.main_caption) - self.amount_of_x)

        self.string_y_average1 = "\nСередні значення функції відгуку\n " + "{:^6}  " * self.N
        self.string_y_average2 = "|{:^5}| " * self.N
        if self.N == 4:
            self.string_b = "\nНатуральні значення факторів\nb0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f}"
            self.string_a = "\nНормовані значення факторів\na0 = {:.3f}, a1 = {:.3f}, a2 = {:.3f}, a3 = {:.3f}"

            self.string_regression_caption = "Y = {:.3f}{:{sign}.3f}*x1{:{sign}.3f}*x2{:{sign}.3f}*x3"
            self.string_regression = "{:.3f}{:{sign}.3f}*({num1:{sign}})" \
                                     "{:{sign}.3f}*({num2:{sign}})" \
                                     "{:{sign}.3f}*({num3:{sign}}) = {res} / {av_y}"
        elif self.N == 8:
            self.string_b_natural = "\nНатуральні значення факторів\n" \
                                    "b0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f}, " \
                                    "b12 = {:.3f}, b13 = {:.3f}, " \
                                    "b23 = {:.3f}, b123 = {:.3f}"
            self.string_b_norm = "\nНормовані значення факторів\n" \
                                 "b0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f}, " \
                                 "b12 = {:.3f}, b13 = {:.3f}, " \
                                 "b23 = {:.3f}, b123 = {:.3f}"
            self.list_of_indexes = ["12", "13", "23", "123"]

            self.string_regression_caption = "Y = {:.3f}{:{sign}.3f}*x1{:{sign}.3f}*x2{:{sign}.3f}*x3" \
                                             "{:{sign}.3f}*x1x2{:{sign}.3f}*x1x3{:{sign}.3f}*x2x3" \
                                             "{:{sign}.3f}*x1x2x3"
            self.string_regression = "{:.3f}{:{sign}.3f}*({num1:{sign}})" \
                                     "{:{sign}.3f}*({num2:{sign}})" \
                                     "{:{sign}.3f}*({num3:{sign}})" \
                                     "{:{sign}.3f}*({num4:{sign}})" \
                                     "{:{sign}.3f}*({num5:{sign}})" \
                                     "{:{sign}.3f}*({num6:{sign}})" \
                                     "{:{sign}.3f}*({num7:{sign}}) = {res} / {av_y}"
        else:
            self.string_b_natural = "\nНатуральні значення факторів\n" \
                                    "b0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f}, " \
                                    "b12 = {:.3f}, b13 = {:.3f}, " \
                                    "b23 = {:.3f}, b123 = {:.3f}, " \
                                    "b11 = {:.3f}, b22 = {:.3f}, b33 = {:.3f}"
            self.list_of_indexes = ["12", "13", "23", "123", "11", "22", "33"]

            self.string_regression_caption = "Y = {:.3f}{:{sign}.3f}*x1{:{sign}.3f}*x2{:{sign}.3f}*x3" \
                                             "{:{sign}.3f}*x1x2{:{sign}.3f}*x1x3{:{sign}.3f}*x2x3" \
                                             "{:{sign}.3f}*x1x2x3{:{sign}.3f}*x1x1{:{sign}.3f}*x2x2" \
                                             "{:{sign}.3f}*x3x3"
            self.string_regression = "{:.3f}{:{sign}.3f}*({num1:{sign}})" \
                                     "{:{sign}.3f}*({num2:{sign}})" \
                                     "{:{sign}.3f}*({num3:{sign}})" \
                                     "{:{sign}.3f}*({num4:{sign}})" \
                                     "{:{sign}.3f}*({num5:{sign}})" \
                                     "{:{sign}.3f}*({num6:{sign}})" \
                                     "{:{sign}.3f}*({num7:{sign}})" \
                                     "{:{sign}.3f}*({num8:{sign}})" \
                                     "{:{sign}.3f}*({num9:{sign}})" \
                                     "{:{sign}.3f}*({num10:{sign}}) = {res} / {av_y}"

        self.string_dispersion1 = "\nПеревірка однорідності за критерієм Кохрена\nДисперсії\n" + "{:^6}  " * self.N
        self.string_dispersion2 = "|{:^5}| " * self.N

        self.string_cohren = "Коефіцієнт Кохрена: {}\nСтупені свободи f1 = {}, f2 = {}\tКритерій Кохрена: {}"
        self.string_cohren_result1 = "Дисперсія однорідна з ймовірністю " + str(self.p)
        self.string_cohren_result2 = "Дисперсія неоднорідна з ймовірністю " + str(self.p)

        self.string_student_marks = "\nПеревірка значущості коефіцієнтів за критерієм Стьюдента\n" \
                                    "Оцінка генеральної дисперсії відтворюваності: {:.3f}, " \
                                    "статистична оцінка дисперсії: {:.3f}"
        if self.N == 4:
            self.string_t = "Коефіцієнти Стьюдента\nt0 = {:.3f}, t1 = {:.3f}, t2 = {:.3f}, t3 = {:.3f}"
        elif self.N == 8:
            self.string_t = "Коефіцієнти Стьюдента\nt0 = {:.3f}, t1 = {:.3f}, t2 = {:.3f}, t3 = {:.3f}, " \
                            "t4 = {:.3f}, t5 = {:.3f}, t6 = {:.3f}, t7 = {:.3f}"
        else:
            self.string_t = "Коефіцієнти Стьюдента\nt0 = {:.3f}, t1 = {:.3f}, t2 = {:.3f}, t3 = {:.3f}, " \
                            "t4 = {:.3f}, t5 = {:.3f}, t6 = {:.3f}, t7 = {:.3f}, t8 = {:.3f}, t9 = {:.3f}, t10 = {:.3f}"

        self.string_student_format = "Ступені свободи f3 = {}\tКритерій Стьюдента: {}"
        self.string_regression_caption_st = "\nТепер рівняння регресії має вигляд\nY = "
        self.string_student_regression = ""

        self.string_fisher_d = "\nПеревірка адекватності за критерієм Фішера\n" \
                               "Кількість значущих коефіцієнтів d = {}, Дисперсія адекватності: {:.3f}"
        self.string_fisher = "Коефіцієнт Фішера: {}\nСтупені свободи f3 = {}, f4 = {}\tКритерій Фішера: {}"
        self.string_fisher_result1 = "Модель адекватна експериментальним даним з ймовірністю " + str(self.p)
        self.string_fisher_result2 = "Модель неадекватна експериментальним даним з ймовірністю " + str(self.p)
        self.string_fisher_result3 = "Неможливо використовувати критерій Фішера тому, що N = d"

        self.matrix_out = copy.deepcopy(self.matrix_x_natural)
        for i in range(self.N):
            for j in range(self.m):
                self.matrix_out[i].append(self.matrix_y[i][j])

    def table_update(self):
        self.main_caption.append("Y" + str(self.m))
        if self.N == 4:
            self.string_main1 += "{:^6}  "
            self.string_main2 += "|{:^5}| "
        elif self.N == 8:
            self.string_main1 += "{:^8}  "
            self.string_main2 += "|{:^7}| "
        else:
            self.string_main1 += "{:^12}"
            self.string_main2 += "|{:^9}| "

    def find_coefficients(self):
        self.out_table()
        if self.N == 4:
            self.list_b = self.coefficients_lineal(self.matrix_x_natural)
            self.out_regression_caption(True)
            self.list_a = self.coefficients_lineal(self.matrix_x_norm)
            self.out_regression_caption(False)
        elif self.N == 8:
            self.list_b_natural = self.coefficients_interaction_squares(self.matrix_x_natural, True)
            self.out_regression_caption(True)
            self.list_b_norm = self.coefficients_interaction_squares(self.matrix_x_norm, False)
            self.out_regression_caption(False)
        else:
            self.list_b_natural = self.coefficients_interaction_squares(self.matrix_x_natural, True)
            self.out_regression_caption(True)

        if self.check_cohren():
            self.check_student()
            self.check_fisher()

    def coefficients_lineal(self, matrix):
        reverse_matrix = list(zip(*matrix))

        self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(self.N)]
        self.m_x = [sum(column) / self.N for column in reverse_matrix]
        self.my = sum(self.average_y) / self.N
        self.list_a_i = [sum([reverse_matrix[row][col] * self.average_y[col] for col in range(self.N)]) / self.N
                         for row in range(self.amount_of_x)]
        self.list_a_ii = [sum([reverse_matrix[row][col] ** 2 for col in range(self.N)]) / self.N for row in
                          range(self.amount_of_x)]

        self.a12, self.a23, self.a31 = 0, 0, 0
        for i in range(self.N):
            self.a12 += reverse_matrix[0][i] * reverse_matrix[1][i]
            self.a23 += reverse_matrix[1][i] * reverse_matrix[2][i]
            self.a31 += reverse_matrix[2][i] * reverse_matrix[0][i]

        self.a12, self.a23, self.a31 = self.a12 / self.N, self.a23 / self.N, self.a31 / self.N
        self.denominator = [[1, self.m_x[0], self.m_x[1], self.m_x[2]],
                            [self.m_x[0], self.list_a_ii[0], self.a12, self.a31],
                            [self.m_x[1], self.a12, self.list_a_ii[1], self.a23],
                            [self.m_x[2], self.a31, self.a23, self.list_a_ii[2]]]

        denominator_det = det(self.denominator)
        self.change = [self.my, self.list_a_i[0], self.list_a_i[1], self.list_a_i[2]]
        reverse_denominator = list(map(list, zip(*self.denominator)))

        list_k = []
        for index in range(len(reverse_denominator)):
            numerator = reverse_denominator[:]
            numerator[index] = self.change
            list_k.append(det(list(zip(*numerator))) / denominator_det)

        return list_k

    def coefficients_interaction_squares(self, matrix, flag):
        #  flag = true, then natural coefficients
        matrix = copy.deepcopy(matrix)
        if flag:
            self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(self.N)]
            for row in range(self.N):
                matrix[row].insert(0, 1)
                matrix[row].append(self.average_y[row])

            matrix_help = []
            matrix_m_ii = []
            reverse_matrix = list(map(list, zip(*matrix)))
            for i in range(len(reverse_matrix) - 1):
                mult = reverse_matrix[i]
                matrix_m_ii.append([])
                for j in range(len(mult)):
                    matrix_help.append([reverse_matrix[col][j] * mult[j] for col in range(len(reverse_matrix))])

                reverse_matrix_m_ii = list(map(list, zip(*matrix_help)))
                for col in range(len(reverse_matrix_m_ii)):
                    matrix_m_ii[i].append(sum(reverse_matrix_m_ii[col]))
                matrix_help = []

            list_k = []
            for row in range(len(matrix_m_ii)):
                list_k.append(matrix_m_ii[row].pop(-1))

            denominator = matrix_m_ii[:]
            denominator_det = det(denominator)

            reverse_det = list(map(list, zip(*denominator)))
            list_b = []
            for i in range(len(reverse_det)):
                numerator = reverse_det[:]
                numerator[i] = list_k
                list_b.append(det(list(zip(*numerator))) / denominator_det)
            return list_b
        else:
            for row in range(self.N):
                matrix[row].insert(0, 1)
            list_b = []
            reverse_matrix = list(map(list, zip(*matrix)))
            for row in range(self.N):
                list_b.append(
                    sum([self.average_y[col] * reverse_matrix[row][col] for col in range(self.N)]) / self.N)
            return list_b

    def add_y(self):
        for i in range(self.N):
            r = random.randint(y_min, y_max + 1)
            self.matrix_y[i].append(r)
            self.matrix_out[i].append(r)
        self.m += 1
        self.table_update()
        self.find_coefficients()

    def check_cohren(self):
        self.dispersion = [sum([(self.matrix_y[row][col] - self.average_y[row]) ** 2
                                for col in range(self.m)]) / self.m
                           for row in range(self.N)]
        self.Coh_coefficient = max(self.dispersion) / sum(self.dispersion)
        self.f1, self.f2 = self.m - 1, self.N
        self.Coh_criterion = get_cohren_criterion(self.f1)
        if self.Coh_coefficient <= self.Coh_criterion:
            flag = True
            self.out_cohren(flag)
            return flag
        else:
            flag = False
            self.out_cohren(flag)
            self.add_y()
            return flag

    def check_student(self):
        if self.N == 4:
            self.list_coefficients = self.list_b
        elif self.N == 8 or self.N == 15:
            self.list_coefficients = self.list_b_natural
        self.d_reproducibility = sum(self.dispersion) / self.N
        self.d_evaluation = self.d_reproducibility /(self.m * self.N)
        self.d_evaluation = math.sqrt(self.d_evaluation)
        reverse_matrix_norm = list(zip(*self.matrix_x_norm_full))
        for col in range(len(reverse_matrix_norm)):
            self.list_beta.append(sum([self.average_y[row] * reverse_matrix_norm[col][row]
                                       for row in range(self.N)]) / self.N)
            self.list_t.append(abs(self.list_beta[col]) / self.d_evaluation)
        self.f3 = self.f1 * self.f2
        self.Stud_criterion = scipy.stats.t.ppf((1 + self.p) / 2, self.f3)
        for index in range(len(self.list_t)):
            if self.list_t[index] >= self.Stud_criterion:
                continue
            else:
                self.list_coefficients[index] = 0
        if self.N == 4:
            self.list_y = [y_lineal_regression(*self.list_coefficients, *self.matrix_x_natural[row])
                           for row in range(self.N)]
        elif self.N == 8:
            self.list_y = [y_interaction_regression(*self.list_coefficients, *self.matrix_x_natural[row])
                           for row in range(self.N)]
        else:
            self.list_y = [y_squares_regression(*self.list_coefficients, *self.matrix_x_natural[row])
                           for row in range(self.N)]

        self.out_student()

    def check_fisher(self):
        self.d = len(self.list_coefficients) - self.list_coefficients.count(0)
        if self.N - self.d == 0:
            self.out_fisher(False)
        else:
            self.d_adequacy =(self.m /(self.N - self.d)) * sum(
                [(self.list_y[row] - self.average_y[row]) ** 2 for row in range(self.N)])
            self.Fish_coefficient = self.d_adequacy / self.d_reproducibility
            self.f4 = self.N - self.d
            self.Fish_criterion = scipy.stats.f.ppf(self.p, self.f4, self.f3)
            if self.Fish_coefficient <= self.Fish_criterion:
                self.check_adequacy = True
                self.out_fisher(True, True)
            else:
                self.out_fisher(True, False)

    def out_table(self):
        print("m =", self.m)
        print(self.string_main1.format(*self.main_caption))
        for row in range(len(self.matrix_out)):
            if row < 9:
                print(self.string_main2.format(str(row + 1) + " ", *self.matrix_out[row]))
            else:
                print(self.string_main2.format(str(row + 1), *self.matrix_out[row]))
        print(self.string_y_average1.format(*self.average_y_caption))
        print(self.string_y_average2.format(*list(map(round, self.average_y))))

    def out_regression_caption(self, flag):
        if self.N == 4:
            if flag:
                print(self.string_b.format(*self.list_b))
                print(self.string_regression_caption.format(*self.list_b, sign="+"))
                self.out_regression(flag)
            else:
                print(self.string_a.format(*self.list_a))
                print(self.string_regression_caption.format(*self.list_a, sign="+"))
                self.out_regression(flag)
        elif self.N == 8:
            if flag:
                print(self.string_b_natural.format(*self.list_b_natural))
                print(self.string_regression_caption.format(*self.list_b_natural, sign="+"))
                self.out_regression(flag)
            else:
                print(self.string_b_norm.format(*self.list_b_norm))
                print(self.string_regression_caption.format(*self.list_b_norm, sign="+"))
                self.out_regression(flag)
        else:
            if flag:
                print(self.string_b_natural.format(*self.list_b_natural))
                print(self.string_regression_caption.format(*self.list_b_natural, sign="+"))
                self.out_regression(flag)

    def out_regression(self, flag=True):
        if self.N == 4:
            if flag:
                matrix, list_k = self.matrix_x_natural, self.list_b
            else:
                matrix, list_k = self.matrix_x_norm, self.list_a
            index = 0
            for x1, x2, x3 in matrix:
                print(self.string_regression.format(*list_k, num1=x1, num2=x2, num3=x3, sign="+",
                                                    res=y_lineal_regression(*list_k, x1, x2, x3),
                                                    av_y=self.average_y[index]))
                index += 1
        elif self.N == 8:
            if flag:
                matrix, list_k = self.matrix_x_natural, self.list_b_natural
            else:
                matrix, list_k = self.matrix_x_norm, self.list_b_norm
            index = 0
            for row in range(len(matrix)):
                x1, x2, x3 = matrix[row][0], matrix[row][1], matrix[row][2]
                x1x2, x1x3, x2x3, x1x2x3 = matrix[row][3], matrix[row][4], matrix[row][5], matrix[row][6]
                print(self.string_regression.format(*list_k, num1=x1, num2=x2, num3=x3,
                                                    num4=x1x2, num5=x1x3, num6=x2x3, num7=x1x2x3, sign="+",
                                                    res=y_interaction_regression(*list_k, x1, x2, x3,
                                                                                 x1x2, x1x3, x2x3, x1x2x3),
                                                    av_y=self.average_y[index]))
                index += 1
        else:
            matrix, list_k = self.matrix_x_natural, self.list_b_natural
            index = 0
            for row in range(len(matrix)):
                x1, x2, x3 = matrix[row][0], matrix[row][1], matrix[row][2]
                x1x2, x1x3, x2x3, x1x2x3 = matrix[row][3], matrix[row][4], matrix[row][5], matrix[row][6]
                x1x1, x2x2, x3x3 = matrix[row][7], matrix[row][8], matrix[row][9]
                print(self.string_regression.format(*list_k, num1=x1, num2=x2, num3=x3,
                                                    num4=x1x2, num5=x1x3, num6=x2x3, num7=x1x2x3,
                                                    num8=x1x1, num9=x2x2, num10=x3x3, sign="+",
                                                    res=y_squares_regression(*list_k, x1, x2, x3,
                                                                             x1x2, x1x3, x2x3, x1x2x3,
                                                                             x1x1, x2x2, x3x3),
                                                    av_y=self.average_y[index]))
                index += 1

    def out_cohren(self, flag):
        print(self.string_dispersion1.format(*self.dispersion_caption))
        print(self.string_dispersion2.format(*list(map(round, self.dispersion))))
        print(self.string_cohren.format(self.Coh_coefficient, self.f1, self.f2, self.Coh_criterion))
        if flag:
            print(self.Coh_coefficient, "<=", self.Coh_criterion)
            print(self.string_cohren_result1)
        else:
            print(self.Coh_coefficient, ">", self.Coh_criterion)
            print(self.string_cohren_result2)

    def out_student(self):
        print(self.string_student_marks.format(self.d_reproducibility, self.d_evaluation))
        print(self.string_t.format(*self.list_t))
        print(self.string_student_format.format(self.f3, self.Stud_criterion))
        count_index = 0
        for index in range(len(self.list_coefficients)):
            if index <= 3:
                if self.list_coefficients[index]:
                    print(self.list_t[index], ">=", self.Stud_criterion,
                          "|b" + str(index) + " - значимий з ймовірністю " + str(self.p))
                else:
                    print(self.list_t[index], "<", self.Stud_criterion,
                          "|b" + str(index) + " - незначимий з ймовірністю " + str(self.p))
            else:
                if self.list_coefficients[index]:
                    print(self.list_t[index], ">=", self.Stud_criterion,
                          "|b" + self.list_of_indexes[count_index] + " - значимий з ймовірністю " + str(self.p))
                else:
                    print(self.list_t[index], "<", self.Stud_criterion,
                          "|b" + self.list_of_indexes[count_index] + " - незначимий з ймовірністю " + str(self.p))
                count_index += 1
        if self.N == 15:
            for elem in self.list_coefficients:
                if elem != 0:
                    list_coefficients.append(elem)
        self.out_regression_student()

    def out_regression_student(self):
        if sum(self.list_coefficients) == 0:
            print(self.string_regression_caption_st + "0")
        elif sum(self.list_coefficients[1:]) == 0:
            print(self.string_regression_caption_st + str(round(self.list_coefficients[0], 3)))
        else:
            if self.list_coefficients[0]:
                self.string_regression_caption_st += "{:.3f}"
                self.string_student_regression += "{:.3f}"
                for i in range(self.N):
                    self.values_regression_st.append([self.list_coefficients[0]])
            else:
                for i in range(self.N):
                    self.values_regression_st.append([])
            count_index = 0
            for i in range(1, len(self.list_coefficients)):
                if i <= 3:
                    if self.list_coefficients[i]:
                        self.string_regression_caption_st += "{:{sign}.3f}*x" + str(i)
                        self.string_student_regression += "{:{sign}.3f}*({:{sign}})"
                elif len(self.list_of_indexes[count_index]) == 2:
                    if self.list_coefficients[i]:
                        self.string_regression_caption_st += "{:{sign}.3f}*x" \
                                                             + self.list_of_indexes[count_index][0] + "x" \
                                                             + self.list_of_indexes[count_index][1]
                        self.string_student_regression += "{:{sign}.3f}*({:{sign}})"
                    count_index += 1
                else:
                    if self.list_coefficients[i]:
                        self.string_regression_caption_st += "{:{sign}.3f}*x" \
                                                             + self.list_of_indexes[count_index][0] + "x" \
                                                             + self.list_of_indexes[count_index][1] + "x" \
                                                             + self.list_of_indexes[count_index][2]
                        self.string_student_regression += "{:{sign}.3f}*({:{sign}})"
                    count_index += 1

            index = 1
            for row in range(self.N):
                for col in range(self.amount_of_x):
                    if self.list_coefficients[index]:
                        self.values_regression_st[row].append(self.list_coefficients[index])
                        self.values_regression_st[row].append(self.matrix_x_natural[row][col])
                    index += 1
                index = 1

            list_out = [self.list_coefficients[i] for i in range(len(self.list_coefficients)) if self.list_coefficients[i] != 0]
            self.string_student_regression += " = {res}"
            print(self.string_regression_caption_st.format(*list_out, sign="+"))
            for row in range(self.N):
                print(self.string_student_regression.format(*self.values_regression_st[row], sign="+",
                                                            res=self.list_y[row]))

    def out_fisher(self, check, flag=True):
        if not check:
            print(self.string_fisher_result3)
        else:
            print(self.string_fisher_d.format(self.d, self.d_adequacy))
            print(self.string_fisher.format(self.Fish_coefficient, self.f3, self.f4, self.Fish_criterion))
            if flag:
                print(self.Fish_coefficient, "<=", self.Fish_criterion)
                print(self.string_fisher_result1)
            else:
                print(self.Fish_coefficient, ">", self.Fish_criterion)
                print(self.string_fisher_result2)


choice = int(input("If you want to check only squares, input 0\n"
                   "If you want to see the whole algorithm, input 1 : "))

if choice:
    for i in range(100):
        while True:
            calc_lineal = Calc(matrix_x_code_1, M, N_lineal)
            if not calc_lineal.check_adequacy:
                calc_interaction = Calc(matrix_x_code_2, M, N_interaction)
                calc_interaction.initial_calculation(calc_lineal.matrix_y)
                if not calc_interaction.check_adequacy:
                    calc_squares = Calc(matrix_x_code_3, M, N_squares)
                    calc_squares.initial_calculation(calc_interaction.matrix_y)
                    if calc_squares.check_adequacy:
                        break
            else:
                break
    if list_coefficients:
        print("Середнє значення усіх значимих коефіцієнтів: ", sum(list_coefficients) / len(list_coefficients))
else:
    for i in range(100):
        calc_squares = Calc(matrix_x_code_3, M, N_squares)
        calc_squares.initial_calculation()
    print("Середнє значення усіх значимих коефіцієнтів: ", sum(list_coefficients)/len(list_coefficients))
