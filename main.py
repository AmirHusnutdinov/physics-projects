from math import comb

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# ===============================================================
#                    ПАРАМЕТРЫ МОДЕЛИ
# ===============================================================
N = 1000  # чисто частиц
L = 1.0  # длина ребра
r = 0.001  # радиус частиц
# phi = N * (4/3) * pi * r^3 формула для нахождения процента газа в нашем кубе если N = 1000 r = 0.01 то это 4%
mass = 1.0
dt = 0.002  # шаг интегрирования
steps_eq = 7000  # время 1 эксперемента ну типа число шагов
steps_stat = 30000  # сколько точек для гистограммы берем

# ===============================================================
#           ИНИЦИАЛИЗАЦИЯ КООРДИНАТ (без пересечений)
# ===============================================================
np.random.seed(42)  # для воспроизводимости
pos = np.random.rand(N, 3) * (L - 2 * r) + r  # отступ от стенок


def ensure_no_overlap(pos, r):
    # pos это список формы (N, 3), где:
    # N — число частиц в системе,
    # 3 — три пространственные координаты: x, y, z.

    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            # просчитываем разность x/y/z-координат между i-ой j-ой частицой
            # высчитываем квадрат расстояния межжу центрами частицы в 3D, но быстро,
            # без необходимости не извлекаем корень
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq < (2 * r) ** 2:
                dist = np.sqrt(dist_sq)  # вот тут мы и посчитали настоящее растояние
                if dist < 1e-12:
                    continue  # проверка на огрничения от железа
                nx, ny, nz = dx / dist, dy / dist, dz / dist  # еденичный напрвленный вектор от j к i
                overlap = 2 * r - dist
                pos[j, 0] += nx * overlap
                pos[j, 1] += ny * overlap
                pos[j, 2] += nz * overlap  # смещаем j-ую частичу относительно i-ой


ensure_no_overlap(pos, r)

# ===============================================================
#      ИНИЦИАЛИЗАЦИЯ СКОРОСТЕЙ ПО МАКСВЕЛЛУ
# ===============================================================
temperature0 = 1.0  # какая-то безразмерная температура
sigma = np.sqrt(temperature0 / mass)  # Это вычисление среднеквадратичного отклонения
# одной компоненты скорости в распределении Максвелла–Больцмана.

vel = np.random.normal(0, sigma, size=(N, 3))
'''Генерирует массив случайных скоростей для всех N частиц.
np.random.normal(loc, scale, size) — генерирует числа из нормального распределения:
loc = 0 — математическое ожидание (среднее значение) = 0.
scale = sigma — стандартное отклонение.
size=(N, 3) — создаётся двумерный массив N строк × 3 столбца:
Каждая строка — вектор скорости одной частицы: [v_x, v_y, v_z].
Всего 3N независимых случайных чисел'''

vel -= np.mean(vel, axis=0)  # убираем дрейф
'''
np.mean(vel, axis=0) vel — массив формы (N, 3).
axis=0 — среднее по первой оси (по частицам), т.е. по строкам.
Результат — вектор длины 3 vel -= ... Вычитает этот вектор из каждой строки массива vel
'''


# ===============================================================
#         УСКОРЕННЫЕ ФУНКЦИИ С JIT (numba)
# ===============================================================

@jit(nopython=True)  # это просто ускорение идет компеляция в машинный код
def step_numba(pos, vel, L, r, mass, dt, momentum):  # momentum суммарный импульс,
    # переданный всем шести стенкам за всё время симуляции

    # --- 1. Движение методом Эйлера---
    pos += vel * dt

    # --- 2. Отражение от стенок + импульс ---
    for i in range(pos.shape[0]):
        for axis in range(3):
            if pos[i, axis] < 0:  # Если частица ушла за левую стенку
                vel[i, axis] *= -1
                pos[i, axis] = 0
                momentum[0] += 2 * mass * abs(vel[i, axis])  # Добавляем в momentum[0] импульс, переданный стенке

            elif pos[i, axis] > L:
                vel[i, axis] *= -1
                pos[i, axis] = L
                momentum[0] += 2 * mass * abs(vel[i, axis])

    # --- 3. Столкновения частиц (O(N²), но JIT-ускорено) ---
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            d2 = 4 * r * r  # (2r)^2
            # снова перебор всех частиц
            if dist_sq < d2:
                dist = np.sqrt(dist_sq)

                inv_dist = 1.0 / (dist + 1e-12)
                # мы избегаем деление на 0 и ускореям код так как деление медленне умножения
                nx, ny, nz = dx * inv_dist, dy * inv_dist, dz * inv_dist

                dvx = vel[i, 0] - vel[j, 0]
                dvy = vel[i, 1] - vel[j, 1]
                dvz = vel[i, 2] - vel[j, 2]
                vn = dvx * nx + dvy * ny + dvz * nz

                if vn < 0:  # движутся навстречу
                    # обновляем скорости (упруго, одинаковые массы)

                    vel[i, 0] -= vn * nx
                    vel[i, 1] -= vn * ny
                    vel[i, 2] -= vn * nz

                    vel[j, 0] += vn * nx
                    vel[j, 1] += vn * ny
                    vel[j, 2] += vn * nz

                    # раздвигаем, чтобы избежать залипания
                    overlap = 2 * r - dist
                    pos[i, 0] += nx * (overlap * 0.5)
                    pos[i, 1] += ny * (overlap * 0.5)
                    pos[i, 2] += nz * (overlap * 0.5)
                    pos[j, 0] -= nx * (overlap * 0.5)
                    pos[j, 1] -= ny * (overlap * 0.5)
                    pos[j, 2] -= nz * (overlap * 0.5)


# ===============================================================
#                   ЭТАП 1: РАВНОВЕСИЕ
# ===============================================================
momentum = np.array([0.0])  # numba требует массив

for step_num in range(steps_eq):
    step_numba(pos, vel, L, r, mass, dt, momentum)
    if step_num % 1000 == 0:
        print(f"Equilibration: {step_num}/{steps_eq}")
# Прогон системы до установления стационарного распределения
# Вывод каждые 1000 шагов — для контроля прогресса


# Вычисляем температуру
# np.sum(..., axis=1) — сумма квадратов по осям → v_i² для каждой частицы
# KE — массив кинетических энергий частиц

KE = 0.5 * mass * np.sum(vel ** 2, axis=1)
# Связь температуры и средней кинетической энергии в 3D
T_model = (2 / 3) * np.mean(KE)
print("\n🔹 Температура:", T_model)

# Давление
# Импульс, переданный всем 6 стенкам: momentum[0].
# Время моделирования: T = steps_eq * dt.
# Суммарная площадь всех стенок: 6 * L² = 6 * area.
# Давление — сила на единицу площади, сила — импульс в единицу времени:
area = L * L
P_model = momentum[0] / (steps_eq * dt * 6 * area)
P_ideal = N * T_model / (L ** 3)
print("🔹 P_model =", P_model)
print("🔹 P_ideal =", P_ideal)
print("🔹 P_model / P_ideal =", P_model / P_ideal)
print("Отношение >1 → система не идеальна по уравнению Ван-дер-Ваальса")

# ===============================================================
#           ЭТАП 2: M7C — ФЛУКТУАЦИИ
# ===============================================================
left_counts = np.empty(steps_stat, dtype=np.int32)
# left_counts — массив для хранения k(t): число частиц с x < L/2 в каждый момент времени.
momentum[0] = 0.0  # сброс импульса

print("\n🔹 Сбор статистики флуктуаций...")

for i in range(steps_stat):
    step_numba(pos, vel, L, r, mass, dt, momentum)
    left_counts[i] = np.sum(pos[:, 0] < L / 2)
    if i % 1000 == 0:
        print(f"  {i}/{steps_stat}", end='\r')

print(f"\n✅ Сбор завершён. Среднее слева: {left_counts.mean():.1f} / {N / 2}")

# ===============================================================
#       ТЕОРИЯ + ГРАФИК
# ===============================================================
values = np.arange(0, N + 1)
binom_dist = np.array([comb(N, k) * 0.5 ** N for k in values])

plt.figure(figsize=(10, 5))
hist, _ = np.histogram(left_counts, bins=np.arange(N + 2), density=True)
plt.step(np.arange(N + 1), hist, where='mid', label="3D Simulation", linewidth=2)

plt.plot(values, binom_dist, 'ro-', markersize=3, label="Binomial theory", alpha=0.7)

# Добавим параметры
mu_th = N / 2
sigma_th = np.sqrt(N / 4)
mu_sim = left_counts.mean()
sigma_sim = left_counts.std()
plt.axvline(mu_th, color='k', linestyle='--', alpha=0.5, label=f"Theory: μ={mu_th:.0f}, σ={sigma_th:.1f}")
plt.axvline(mu_sim, color='b', linestyle=':', alpha=0.7, label=f"Sim: μ={mu_sim:.1f}, σ={sigma_sim:.1f}")
# σ - это мера разброса — насколько сильно значения k отклоняются от среднего.
plt.xlabel("Число частиц в левой половине")
plt.ylabel("Вероятность")
plt.title(f"Флуктуации числа частиц (N={N}, r={r})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
