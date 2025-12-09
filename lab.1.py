import math

EPS = 1e-9


# ---------- Разбор входного файла ----------

def read_lp_from_file(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    sense = lines[0].lower()    # max / min
    n, m = map(int, lines[1].split())
    c = list(map(float, lines[2].split()))

    constraints = []
    for k in range(m):
        parts = lines[3 + k].split()
        # последние два токена: знак и правая часть
        rel = parts[-2]         # <=, >=, =
        b = float(parts[-1])
        a = list(map(float, parts[:-2]))
        constraints.append((a, rel, b))

    return sense, c, constraints


# ---------- Базовый симплекс для задачи max ----------

def simplex_max(tableau, basis, eps=EPS, max_iter=100):
    """
    tableau: список строк [a_1 ... a_n | b], последняя строка - строка целевой функции
    basis: список индексов базисных переменных (по столбцам)
    """
    m = len(tableau) - 1      # число ограничений
    n = len(tableau[0]) - 1   # число переменных (столбцов без правой части)

    it = 0
    while True:
        it += 1
        if it > max_iter:
            raise RuntimeError("Превышено число итераций симплекс-метода")

        obj = tableau[-1]

        # Выбираем входящую переменную: столбец с максимальным положительным коэффициентом
        entering = None
        max_val = eps
        for j in range(n):
            if obj[j] > max_val:
                max_val = obj[j]
                entering = j

        # Если нечего вводить в базис — оптимум найден
        if entering is None:
            break

        # Минимальное отношение b_i / a_ij для a_ij > 0
        leaving = None
        min_ratio = float("inf")
        for i in range(m):
            a_ij = tableau[i][entering]
            if a_ij > eps:
                ratio = tableau[i][-1] / a_ij
                if ratio < min_ratio - eps:
                    min_ratio = ratio
                    leaving = i

        if leaving is None:
            raise ValueError("Целевая функция неограничена (unbounded)")

        # Поворот (pivot)
        pivot = tableau[leaving][entering]

        # Нормируем ведущую строку
        for j in range(n + 1):
            tableau[leaving][j] /= pivot

        # Обнуляем столбец entering в остальных строках
        for i in range(m + 1):
            if i == leaving:
                continue
            factor = tableau[i][entering]
            if abs(factor) > eps:
                for j in range(n + 1):
                    tableau[i][j] -= factor * tableau[leaving][j]

        basis[leaving] = entering

    # Формируем решение
    solution = [0.0] * n
    for i, bj in enumerate(basis):
        solution[bj] = tableau[i][-1]

    value = tableau[-1][-1]
    return solution, value


# ---------- Двухфазный симплекс ----------

def two_phase_simplex(c, constraints, debug=False):
    """
    c         – коэффициенты целевой функции (для max)
    constraints – список (a, rel, b), rel ∈ {'<=','>=','='}
    Возвращает: (x_opt, Z_opt, error_message_or_None)
    """

    # 1) Нормируем b, чтобы b >= 0
    normalized = []
    for a, rel, b in constraints:
        a = list(a)
        if b < 0:
            a = [-x for x in a]
            if rel == "<=":
                rel = ">="
            elif rel == ">=":
                rel = "<="
            b = -b
        normalized.append((a, rel, b))
    constraints = normalized

    m = len(constraints)
    n = len(c)

    # Матрица ограничений и типы переменных
    A = []
    b_vec = []
    basis = []
    var_types = ["x"] * n      # первые n – исходные x

    for a, rel, b in constraints:
        row = list(a)
        # выравниваем длину строки под текущее число переменных
        while len(row) < len(var_types):
            row.append(0.0)

        if rel == "<=":
            # slack-переменная
            s_idx = len(var_types)
            var_types.append("slack")
            for r in A:
                r.append(0.0)
            row.append(1.0)
            basis.append(s_idx)

        elif rel == ">=":
            # избыточная (с коэффициентом -1) + искусственная
            s_idx = len(var_types)
            var_types.append("slack")
            for r in A:
                r.append(0.0)
            row.append(-1.0)

            a_idx = len(var_types)
            var_types.append("art")
            for r in A:
                r.append(0.0)
            row.append(1.0)
            basis.append(a_idx)

        elif rel == "=":
            # только искусственная переменная
            a_idx = len(var_types)
            var_types.append("art")
            for r in A:
                r.append(0.0)
            row.append(1.0)
            basis.append(a_idx)
        else:
            raise ValueError("Неверный знак ограничения")

        A.append(row)
        b_vec.append(b)

    total_vars = len(var_types)
    # добиваем строки A до полной длины
    for r in A:
        if len(r) < total_vars:
            r.extend([0.0] * (total_vars - len(r)))

    has_artificial = any(t == "art" for t in var_types)

    # ---------- Фаза 1: поиск допустимого решения ----------

    if has_artificial:
        # строим начальный симплекс-табло для фазы 1
        tableau = [row[:] + [b] for row, b in zip(A, b_vec)]
        obj = [0.0] * (total_vars + 1)

        # целевая для фазы 1: max(-sum a_j)
        for j, t in enumerate(var_types):
            if t == "art":
                obj[j] = -1.0
        tableau.append(obj)

        # приводим к каноническому виду (обнуляем коэффициенты при базисных искусственных)
        for i, bj in enumerate(basis):
            if var_types[bj] == "art":
                for j in range(total_vars + 1):
                    tableau[-1][j] += tableau[i][j]

        if debug:
            print("Phase 1 initial tableau:")
            for row in tableau:
                print([round(x, 3) for x in row])

        sol1, val1 = simplex_max(tableau, basis)

        if val1 < -1e-7:
            return None, None, "Задача не имеет допустимых решений (несовместна)"

        # удаляем искусственные переменные из табло
        m1 = len(tableau) - 1
        n_tot = len(tableau[0]) - 1
        art_indices = [j for j, t in enumerate(var_types) if t == "art"]

        # стараемся выведать искусственные из базиса
        for i, bj in enumerate(basis):
            if bj in art_indices:
                pivot_col = None
                for j in range(n_tot):
                    if var_types[j] != "art" and abs(tableau[i][j]) > EPS:
                        pivot_col = j
                        break
                if pivot_col is not None:
                    pivot = tableau[i][pivot_col]
                    for j in range(n_tot + 1):
                        tableau[i][j] /= pivot
                    for r in range(m1 + 1):
                        if r == i:
                            continue
                        factor = tableau[r][pivot_col]
                        if abs(factor) > EPS:
                            for j in range(n_tot + 1):
                                tableau[r][j] -= factor * tableau[i][j]
                    basis[i] = pivot_col

        # реально выкидываем столбцы искусственных
        keep_cols = [j for j in range(n_tot) if var_types[j] != "art"]
        new_index = {j: k for k, j in enumerate(keep_cols)}
        new_tableau = []
        for r in range(m1 + 1):
            new_row = [tableau[r][j] for j in keep_cols] + [tableau[r][-1]]
            new_tableau.append(new_row)
        tableau = new_tableau
        var_types = [var_types[j] for j in keep_cols]
        basis = [new_index[bj] for bj in basis]

    else:
        tableau = [row[:] + [b] for row, b in zip(A, b_vec)]
        tableau.append([0.0] * (len(A[0]) + 1))

    # ---------- Фаза 2: оптимизация исходной целевой функции ----------

    total_vars = len(var_types)
    c_ext = list(c) + [0.0] * (total_vars - len(c))

    # строка целевой функции: r_j = c_j - sum(c_B * a_Bj),  Z = sum(c_B * b_B)
    obj = [0.0] * (total_vars + 1)
    for j in range(total_vars):
        obj[j] = c_ext[j]
    obj[-1] = 0.0

    for i, bj in enumerate(basis):
        coeff = c_ext[bj]
        if abs(coeff) > EPS:
            for j in range(total_vars + 1):
                obj[j] -= coeff * tableau[i][j]

    tableau[-1] = obj

    if debug:
        print("Phase 2 initial tableau:")
        for row in tableau:
            print([round(x, 3) for x in row])

    sol2, val2 = simplex_max(tableau, basis)

    # В нашем представлении val2 = -Z*, поэтому берём с минусом
    x_opt = sol2[:len(c)]
    z_opt = -val2

    return x_opt, z_opt, None


# ---------- Обёртка для задачи min / max и вывод результата ----------

def solve_lp_from_file(input_file: str, output_file: str = "lp_output.txt", debug=False):
    sense, c, constraints = read_lp_from_file(input_file)

    # если задача на минимум — домножаем целевую на -1 и решаем как max
    if sense == "min":
        c = [-ci for ci in c]

    x_opt, z_opt, err = two_phase_simplex(c, constraints, debug=debug)

    with open(output_file, "w", encoding="utf-8") as f:
        if err is not None:
            print(err)
            f.write(err + "\n")
        else:
            if sense == "min":
                z_opt = -z_opt
            print("Оптимальная точка:", x_opt)
            print("Значение целевой функции:", z_opt)

            f.write("Оптимальная точка: " + ", ".join(f"{xi:.6g}" for xi in x_opt) + "\n")
            f.write(f"Оптимальное значение целевой функции: {z_opt:.6g}\n")


if __name__ == "__main__":
    # пример запуска: читаем lp_input.txt, пишем ответ в lp_output.txt
    solve_lp_from_file("lp_input.txt", "lp_output.txt", debug=False)
