import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Физические параметры
g = 9.81  # ускорение свободного падения, м/с^2
m = 1.0  # масса шара, кг
R = 0.1  # радиус шара, м
I = (2 / 5) * m * R**2  # момент инерции однородного шара
mu = 0.2  # коэффициент трения скольжения

# Угол наклона плоскости (в радианах)
theta = np.radians(15)  # 15 градусов

# Начальные условия
x0 = 0.0  # начальная координата центра масс по оси x (вдоль наклона)
v0 = 0.0  # начальная скорость поступательного движения
omega0 = 0.0  # начальная угловая скорость
y0 = [x0, v0, omega0]

# Время интегрирования
t_span = (0, 5)  # от 0 до 5 секунд
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Флаг режима: 'incline' или 'horizontal'
mode = "incline"


def equations(t, y):
    x, v, omega = y

    if mode == "incline":
        # Сила тяжести вдоль наклонной плоскости
        F_grav = m * g * np.sin(theta)
        N = m * g * np.cos(theta)  # нормальная сила
    else:
        # Горизонтальная плоскость
        F_grav = 0.0
        N = m * g

    # Сила трения (предельная)
    F_fr_max = mu * N

    # Скорость точки контакта (v - omega*R)
    v_rel = v - omega * R

    # Определяем направление и величину силы трения
    if abs(v_rel) < 1e-6:  # условно без проскальзывания
        # Попытаемся найти силу трения, обеспечивающую чистое качение
        # Уравнения движения:
        # m * a = F_grav - F_f
        # I * alpha = F_f * R
        # При чистом качении: a = alpha * R  =>  a = (F_f * R^2) / I
        # Подставляем: m * (F_f * R^2 / I) = F_grav - F_f
        # => F_f * (m * R^2 / I + 1) = F_grav
        denom = m * R**2 / I + 1
        F_f_needed = F_grav / denom if denom != 0 else 0.0

        if abs(F_f_needed) <= F_fr_max:
            # Сила трения покоя достаточна — чистое качение
            F_f = F_f_needed
        else:
            # Недостаточно силы трения — проскальзывание
            F_f = (
                -F_fr_max * np.sign(v_rel)
                if v_rel != 0
                else -F_fr_max * np.sign(F_grav)
            )
    else:
        # Есть проскальзывание — сила трения скольжения
        F_f = -F_fr_max * np.sign(v_rel)

    # Ускорения
    a = (F_grav - F_f) / m
    alpha = (F_f * R) / I

    return [v, a, alpha]


# Интегрируем уравнения движения
sol = solve_ivp(
    equations, t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10
)

t = sol.t
x, v, omega = sol.y

# Энергии
if mode == "incline":
    h = x * np.sin(theta)
    U = m * g * h
else:
    U = np.zeros_like(t)
K_trans = 0.5 * m * v**2
K_rot = 0.5 * I * omega**2
E_total = U + K_trans + K_rot

# Визуализация движения (анимация)
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.1, max(x) + 0.2)
if mode == "incline":
    # Рисуем наклонную плоскость
    incline_length = max(x) + 0.3
    x_line = np.array([0, incline_length])
    y_line = np.array([0, -incline_length * np.tan(theta)])
    ax.plot(x_line, y_line, "k-", lw=3)
    ax.set_ylim(-incline_length * np.tan(theta) - 0.15, 0.15)
else:
    ax.set_ylim(-0.15, 0.15)
    ax.axhline(0, color="k", lw=3)

(ball,) = ax.plot([], [], "o", color="red", markersize=20)
(trace,) = ax.plot([], [], "-", color="gray", alpha=0.5)

history_x = []
history_y = []


def animate(i):
    xi = x[i]
    if mode == "incline":
        yi = -xi * np.tan(theta) + R / np.cos(theta)
    else:
        yi = R
    ball.set_data([xi], [yi])
    history_x.append(xi)
    history_y.append(yi)
    trace.set_data(history_x, history_y)
    return ball, trace


ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True, repeat=False)

plt.title(
    f'Качение шара по {"наклонной" if mode=="incline" else "горизонтальной"} плоскости'
)
plt.xlabel("x (м)")
plt.ylabel("y (м)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Проверка сохранения энергии
plt.figure(figsize=(10, 6))
plt.plot(t, E_total, label="Полная энергия")
plt.plot(t, K_trans, label="Кинетическая (поступательная)")
plt.plot(t, K_rot, label="Кинетическая (вращательная)")
if mode == "incline":
    plt.plot(t, U, label="Потенциальная")
plt.xlabel("Время (с)")
plt.ylabel("Энергия (Дж)")
plt.title("Энергии в процессе движения")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Анализ режима: проскальзывание или нет
v_rel = v - omega * R
plt.figure(figsize=(10, 4))
plt.plot(t, v_rel, label="Скорость проскальзывания $v - \\omega R$")
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.xlabel("Время (с)")
plt.ylabel("Скорость (м/с)")
plt.title("Скорость относительного проскальзывания")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Печать информации о наличии проскальзывания
if np.any(np.abs(v_rel) > 1e-4):
    print("Обнаружено проскальзывание в некоторые моменты времени.")
else:
    print("Шар катится без проскальзывания всё время.")

# Момент импульса относительно точки контакта
# L = I*omega + m*v*R (для горизонтальной плоскости и центра масс)
# Для наклонной плоскости точка контакта движется, поэтому лучше смотреть относительно центра масс:
L_cm = I * omega
plt.figure(figsize=(10, 4))
plt.plot(t, L_cm, label="Момент импульса относительно центра масс")
plt.xlabel("Время (с)")
plt.ylabel("Момент импульса (кг·м²/с)")
plt.title("Момент импульса шара")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
