import random
import numpy as np
import copy
import math

x_min = [-25, -20, -25]
x_max = [75, 60, -10]

x_average_min = sum(x_min) / 3
x_average_max = sum(x_max) / 3

y_min = 200 + int(x_average_min)
y_max = 200 + int(x_average_max)

M = 3
N = 4

cohren_values = {1: 9065, 2: 7679, 3: 6841, 4: 6287, 5: 5892, 6: 5598, 7: 5365, 8: 5175, 9: 5017,
                 10: 4884, range(11, 17): 4366, range(17, 37): 3720, range(37, 145): 3093}

student_values = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
                  21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
                  26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

fisher_values = {8: {1: 5.3, 2: 4.5, 3: 4.1, 4: 3.8}, 9: {1: 5.1, 2: 4.3, 3: 3.9, 4: 3.6},
                 10: {1: 5.0, 2: 4.1, 3: 3.7, 4: 3.5}, 11: {1: 4.8, 2: 4.0, 3: 3.6, 4: 3.4},
                 12: {1: 4.8, 2: 3.9, 3: 3.5, 4: 3.3}, 13: {1: 4.7, 2: 3.8, 3: 3.4, 4: 3.2},
                 14: {1: 4.6, 2: 3.7, 3: 3.3, 4: 3.1}, 15: {1: 4.4, 2: 3.7, 3: 3.3, 4: 3.1},
                 16: {1: 4.5, 2: 3.6, 3: 3.2, 4: 3.0}, 17: {1: 4.5, 2: 3.6, 3: 3.2, 4: 3.0},
                 18: {1: 4.4, 2: 3.6, 3: 3.2, 4: 2.9}, 19: {1: 4.4, 2: 3.5, 3: 3.1, 4: 2.9},
                 range(20, 22): {1: 4.4, 2: 3.5, 3: 3.1, 4: 2.9}, range(22, 24): {1: 4.3, 2: 3.4, 3: 3.1, 4: 2.8},
                 range(24, 26): {1: 4.3, 2: 3.4, 3: 3.0, 4: 2.8}, range(26, 28): {1: 4.2, 2: 3.4, 3: 3.0, 4: 2.7},
                 range(28, 30): {1: 4.2, 2: 3.3, 3: 3.0, 4: 2.7}, range(30, 40): {1: 4.2, 2: 3.3, 3: 2.9, 4: 2.7},
                 range(40, 60): {1: 4.1, 2: 3.2, 3: 2.9, 4: 2.6}, range(60, 120): {1: 4.0, 2: 3.2, 3: 2.8, 4: 2.5}
                 }


def det(arr):
    return np.linalg.det(np.array(arr))


def y(k0=0, k1=0, k2=0, k3=0, x1=0, x2=0, x3=0):
    return k0 + k1 * x1 + k2 * x2 + k3 * x3


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


def get_student_criterion(f3):
    for key in student_values.keys():
        if f3 == key:
            value = student_values.get(key)
            break
    else:
        value = 1.960

    return value


def get_fisher_criterion(f3, f4):
    value = 0
    row_last = {1: 3.8, 2: 3.0, 3: 2.6, 4: 2.4}
    for key_1 in fisher_values:
        if type(key_1) == int:
            if f3 == key_1:
                for key_2 in fisher_values.get(key_1):
                    if f4 == key_2:
                        value = fisher_values.get(key_1).get(key_2)
                        break
                if value:
                    break
        else:
            if f3 in key_1:
                for key_2 in fisher_values.get(key_1):
                    if f4 == key_2:
                        value = fisher_values.get(key_1).get(key_2)
                        break
                if value:
                    break
    else:
        for key in row_last:
            if f4 == key:
                value = row_last.get(key)
                break
    return value


matrix_x_code = [[-1, -1, -1],
                 [-1, +1, +1],
                 [+1, -1, +1],
                 [+1, +1, -1]]


class Calc:
    def __init__(self, main_matrix, m):
        self.matrix_x_code, self.matrix_x_natural, self.matrix_x_norm, self.matrix_y = main_matrix, [], [], []
        self.matrix_x_norm_full = []
        self.average_y, self.m_x, self.list_a_i, self.list_a_ii = None, None, None, None
        self.amount_of_x, self.my, self.a12, self.a23, self.a31 = None, None, None, None, None
        self.denominator, self.change, self.list_b, self.list_a = None, None, None, None
        self.list_delta, self.list_x0 = [], []
        self.m, self.p = m, 0.95
        self.p = 0.95
        self.dispersion, self.Coh_coefficient, self.Coh_criterion = None, None, None
        self.f1, self.f2, self.f3, self.f4 = None, None, None, None
        self.d_reproducibility, self.d_evaluation = None, None
        self.list_beta, self.list_t, self.Stud_criterion = [], [], None
        self.d, self.d_adequacy, self.Fish_coefficient, self.Fish_criterion = None, None, None, None
        self.list_y = None

        self.main_caption, self.average_y_caption = ["X1", "X2", "X3"], ["Y1", "Y2", "Y3", "Y4"]
        self.dispersion_caption = ["D1", "D2", "D3", "D4"]
        for i in range(self.m):
            self.main_caption.append("Y" + str(i + 1))
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
        self.initial_calculation()

    def initial_calculation(self):
        self.amount_of_x = len(self.matrix_x_code[0])
        self.matrix_y = [[random.randint(y_min, y_max + 1) for j in range(self.m)] for i in range(N)]
        self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(N)]

        for i in range(self.amount_of_x):
            self.list_delta.append(abs(x_max[i] - x_min[i]) / 2)
            self.list_x0.append((x_max[i] + x_min[i]) / 2)

        for i in range(N):
            self.matrix_x_natural.append([])
            self.matrix_x_norm.append([])
            for j in range(self.amount_of_x):
                self.matrix_x_natural[i].append(x_min[j]) if matrix_x_code[i][j] == -1 \
                    else self.matrix_x_natural[i].append(x_max[j])
                self.matrix_x_norm[i].append((self.matrix_x_natural[i][j] - self.list_x0[j]) /(self.list_delta[j]))

        self.matrix_x_norm_full = copy.deepcopy(self.matrix_x_norm)
        for i in range(N):
            self.matrix_x_norm_full[i].insert(0, 1)
        self.initial_format()
        self.find_coefficients()

    def initial_format(self):
        self.string_main1 = "№ " + "{:^6}  " * self.amount_of_x + "{:^6}  " * (
                    len(self.main_caption) - self.amount_of_x)
        self.string_main2 = "{} " + "| {:+} | " * self.amount_of_x + "|{:^5}| " * (
                    len(self.main_caption) - self.amount_of_x)

        self.string_y_average1 = "\nСередні значення функції відгуку\n " + "{:^6}  " * N
        self.string_y_average2 = "|{:^5}| "*N

        self.string_b = "\nНатуральні значення факторів\nb0 = {:.3f}, b1 = {:.3f}, b2 = {:.3f}, b3 = {:.3f}"
        self.string_a = "\nНормовані значення факторів\na0 = {:.3f}, a1 = {:.3f}, a2 = {:.3f}, a3 = {:.3f}"

        self.string_regression_caption = "Y = {:.3f}{:{sign}.3f}*x1{:{sign}.3f}*x2{:{sign}.3f}*x3"
        self.string_regression = "{:.3f}{:{sign}.3f}*({num1:{sign}})" \
                                 "{:{sign}.3f}*({num2:{sign}})" \
                                 "{:{sign}.3f}*({num3:{sign}}) = {res} / {av_y}"

        self.string_dispersion1 = "\nПеревірка однорідності за критерієм Кохрена\nДисперсії\n " + "{:^6}  " * N
        self.string_dispersion2 = "|{:^5}| "*N

        self.string_cohren = "Коефіцієнт Кохрена: {}\nСтупені свободи f1 = {}, f2 = {}\tКритерій Кохрена: {}"
        self.string_cohren_result1 = "Дисперсія однорідна з ймовірністю " + str(self.p)
        self.string_cohren_result2 = "Дисперсія неоднорідна з ймовірністю " + str(self.p)

        self.string_student_marks = "\nПеревірка значущості коефіцієнтів за критерієм Стьюдента\n" \
                                    "Оцінка генеральної дисперсії відтворюваності: {:.3f}, " \
                                    "статистична оцінка дисперсії: {:.3f}"
        self.string_t = "Коефіцієнти Стьюдента\nt0 = {:.3f}, t1 = {:.3f}, t2 = {:.3f}, t3 = {:.3f}"
        self.string_student_format = "Ступені свободи f3 = {}\tКритерій Стьюдента: {}"
        self.string_regression_caption_st = "Тепер рівняння регресії має вигляд\nY = "
        self.string_student_regression = ""

        self.string_fisher_d = "\nПеревірка адекватності за критерієм Фішера\n" \
                               "Кількість значущих коефіцієнтів d = {}, Дисперсія адекватності: {:.3f}"
        self.string_fisher = "Коефіцієнт Фішера: {}\nСтупені свободи f3 = {}, f4 = {}\tКритерій Фішера: {}"
        self.string_fisher_result1 = "Модель адекватна експериментальним даним з ймовірністю " + str(self.p)
        self.string_fisher_result2 = "Модель неадекватна експериментальним даним з ймовірністю " + str(self.p)
        self.string_fisher_result3 = "Неможливо використовувати критерій Фішера тому, що N = d"

        self.matrix_out = copy.deepcopy(self.matrix_x_natural)
        for i in range(N):
            for j in range(self.m):
                self.matrix_out[i].append(self.matrix_y[i][j])

    def table_update(self):
        self.main_caption.append("Y" + str(self.m))
        self.string_main1 += "{:^6}  "
        self.string_main2 += "|{:^5}| "

    def find_coefficients(self):
        self.out_table()
        self.list_b = self.coefficients(self.matrix_x_natural)
        self.out_regression_caption(True)
        self.list_a = self.coefficients(self.matrix_x_norm)
        self.out_regression_caption(False)
        if self.check_cohren():
            self.check_student()
            self.check_fisher()

    def coefficients(self, matrix):
        reverse_matrix = list(zip(*matrix))

        self.average_y = [sum(self.matrix_y[i]) / self.m for i in range(N)]
        self.m_x = [sum(column) / N for column in reverse_matrix]
        self.my = sum(self.average_y) / N
        self.list_a_i = [sum([reverse_matrix[row][col] * self.average_y[col] for col in range(N)]) / N
                         for row in range(self.amount_of_x)]
        self.list_a_ii = [sum([reverse_matrix[row][col] ** 2 for col in range(N)]) / N for row in
                          range(self.amount_of_x)]

        self.a12, self.a23, self.a31 = 0, 0, 0
        for i in range(N):
            self.a12 += reverse_matrix[0][i] * reverse_matrix[1][i]
            self.a23 += reverse_matrix[1][i] * reverse_matrix[2][i]
            self.a31 += reverse_matrix[2][i] * reverse_matrix[0][i]

        self.a12, self.a23, self.a31 = self.a12 / 4, self.a23 / 4, self.a31 / 4
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

    def add_y(self):
        for i in range(N):
            r = random.randint(y_min, y_max + 1)
            self.matrix_y[i].append(r)
            self.matrix_out[i].append(r)
        self.m += 1
        self.table_update()
        self.find_coefficients()

    def check_cohren(self):
        self.dispersion = [sum([(self.matrix_y[row][col] - self.average_y[row]) ** 2
                           for col in range(self.m)]) / self.m
                           for row in range(N)]
        self.Coh_coefficient = max(self.dispersion) / sum(self.dispersion)
        self.f1, self.f2 = self.m - 1, N
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
        self.d_reproducibility = sum(self.dispersion) / N
        self.d_evaluation = self.d_reproducibility /(self.m * N)
        self.d_evaluation = math.sqrt(self.d_evaluation)
        reverse_matrix_norm = list(zip(*self.matrix_x_norm_full))
        for col in range(N):
            self.list_beta.append(sum([self.average_y[row] * reverse_matrix_norm[col][row] for row in range(N)]) / N)
            self.list_t.append(abs(self.list_beta[col]) / self.d_evaluation)
        self.f3 = self.f1 * self.f2
        self.Stud_criterion = get_student_criterion(self.f3)
        for index in range(N):
            if self.list_t[index] >= self.Stud_criterion:
                continue
            else:
                self.list_b[index] = 0
        self.list_y = [y(*self.list_b, *self.matrix_x_natural[row]) for row in range(N)]
        self.out_student()

    def check_fisher(self):
        self.d = N - self.list_b.count(0)
        if N - self.d == 0:
            self.out_fisher(False)
        else:
            self.d_adequacy =(self.m /(N - self.d)) * sum(
                [(self.list_y[row] - self.average_y[row]) ** 2 for row in range(N)])
            self.Fish_coefficient = self.d_adequacy / self.d_reproducibility
            self.f4 = N - self.d
            self.Fish_criterion = get_fisher_criterion(self.f3, self.f4)
            if self.Fish_coefficient <= self.Fish_criterion:
                self.out_fisher(True, True)
            else:
                self.out_fisher(True, False)

    def out_table(self):
        print("m =", self.m)
        print(self.string_main1.format(*self.main_caption))
        for row in range(len(self.matrix_out)):
            print(self.string_main2.format(row + 1, *self.matrix_out[row]))
        print(self.string_y_average1.format(*self.average_y_caption))
        print(self.string_y_average2.format(*list(map(round, self.average_y))))

    def out_regression_caption(self, flag):
        if flag:
            print(self.string_b.format(*self.list_b))
            print(self.string_regression_caption.format(*self.list_b, sign="+"))
            self.out_regression(flag)
        else:
            print(self.string_a.format(*self.list_a))
            print(self.string_regression_caption.format(*self.list_a, sign="+"))
            self.out_regression(flag)

    def out_regression(self, flag):
        if flag:
            matrix, list_k = self.matrix_x_natural, self.list_b
        else:
            matrix, list_k = self.matrix_x_norm, self.list_a
        index = 0
        for x1, x2, x3 in matrix:
            print(self.string_regression.format(*list_k, num1=x1, num2=x2, num3=x3, sign="+",
                  res=y(*list_k, x1, x2, x3), av_y=self.average_y[index]))
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
        for index in range(N):
            if self.list_b[index]:
                print(self.list_t[index], ">=", self.Stud_criterion,
                      "|b" + str(index) + " - значимий з ймовірністю " + str(self.p))
            else:
                print(self.list_t[index], "<", self.Stud_criterion,
                      "|b" + str(index) + " - незначимий з ймовірністю " + str(self.p))
        self.out_regression_student()

    def out_regression_student(self):
        if sum(self.list_b) == 0:
            print(self.string_regression_caption_st + "0")
        elif sum(self.list_b[1:]) == 0:
            print(self.string_regression_caption_st + str(round(self.list_b[0], 3)))
        else:
            if self.list_b[0]:
                self.string_regression_caption_st += "{:.3f}"
                self.string_student_regression += "{:.3f}"
                for i in range(N):
                    self.values_regression_st.append([self.list_b[0]])
            else:
                for i in range(N):
                    self.values_regression_st.append([])

            for i in range(1, N):
                if self.list_b[i]:
                    self.string_regression_caption_st += "{:{sign}.3f}*x" + str(i)
                    self.string_student_regression +="{:{sign}.3f}*({:{sign}})"
                else:
                    continue

            index = 1
            for row in range(N):
                for col in range(self.amount_of_x):
                    if self.list_b[index]:
                        self.values_regression_st[row].append(self.list_b[index])
                        self.values_regression_st[row].append(self.matrix_x_natural[row][col])
                    index+=1
                index = 1

            list_out = [self.list_b[i] for i in range(N) if self.list_b[i] != 0]
            self.string_student_regression += " = {res}"
            print(self.string_regression_caption_st.format(*list_out, sign="+"))
            for row in range(N):
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


calc = Calc(matrix_x_code, M)
