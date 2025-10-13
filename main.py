"""

Реализация задачи "Полёт камня" с линейным и квадратичным сопротивлением.
Конфигурация параметров в начале файла.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# -------------------------
# Конфигурация (меняйте здесь)
# -------------------------
m = 0.2  # масса, кг
g = 9.81  # m/s^2
v0 = 30.0  # начальная скорость, м/с
angle_deg = 40.0  # угол броска, градусы
model = 'both'  # 'none', 'linear', 'quadratic', 'both' (both = рисует всё для сравнения)

# Линейная модель: F = -b v
b = 0.1  # коэффициент сопротивления (Н·с/м). Эффективный; b/m = k с^{-1}

# Квадратичная модель: F = -c v |v|
c = 0.005  # коэффициент (примерно 0.5 * rho * Cd * A)

# Решение ОДУ: параметры временного шага и максимум времени
t_max = 30.0
rtol = 1e-8
atol = 1e-10

# Серия опытов (необязательно) -- уберите или добавьте значения, если хотите прогнать разные параметры
series = [
    {'v0': v0, 'angle_deg': angle_deg, 'label': 'базовый'},
    # пример: {'v0': 40, 'angle_deg': 30, 'label': 'быстрый, малый угол'},
]


# -------------------------
# Функции: Правая часть для разных моделей
# -------------------------
def rhs_none(_, state):
    """Без сопротивления"""
    x, y, vx, vy = state
    ax = 0.0
    ay = -g
    return [vx, vy, ax, ay]


def rhs_linear(_, state, b_coef):
    x, y, vx, vy = state
    k = b_coef / m
    v = np.hypot(vx, vy)
    # линейная модель по компонентам:
    ax = -k * vx
    ay = -g - k * vy
    return [vx, vy, ax, ay]


def rhs_quadratic(_, state, c_coef):
    x, y, vx, vy = state
    v = np.hypot(vx, vy)
    if v == 0.0:
        ax = 0.0
        ay = -g
    else:
        ax = - (c_coef / m) * v * vx
        ay = -g - (c_coef / m) * v * vy
    return [vx, vy, ax, ay]


# -------------------------
# Аналитические формулы
# -------------------------
def analytic_no_drag(v0, angle_rad):
    """Классический случай без сопротивления"""
    v0x = v0 * np.cos(angle_rad)
    v0y = v0 * np.sin(angle_rad)
    t_fall = 2.0 * v0y / g
    range_ = v0x * t_fall
    h_max = v0y ** 2 / (2.0 * g)
    return {'t_fall': t_fall, 'range': range_, 'h_max': h_max}


def analytic_linear(v0, angle_rad, b_coef):
    """
    Выражения для скоростей и положений при линейном сопротивлении F=-b v.
    k = b/m.
    vx(t) = v0x * exp(-k t)
    x(t) = v0x / k * (1 - exp(-k t))
    vy(t) = (v0y + g/k) * exp(-k t) - g/k
    y(t) = (v0y + g/k)/k * (1 - exp(-k t)) - g*t/k
    (приведен вид, который можно вычислить численно)
    """
    k = b_coef / m
    v0x = v0 * np.cos(angle_rad)
    v0y = v0 * np.sin(angle_rad)
    return {'k': k, 'v0x': v0x, 'v0y': v0y}


# -------------------------
# Поиск точки падения более точно (интерполяция линейная на последнем сегменте)
# -------------------------
def find_landing_time_and_x(sol):
    t = sol.t
    y = sol.y[1]
    x = sol.y[0]
    if np.all(y >= 0):
        # не упал в пределах tspan
        return None
    # найдём последний индекс где y>0
    idx = np.where(y >= 0)[0]
    if len(idx) == 0:
        # стартовали под землёй
        return t[0], x[0]
    i = idx[-1]
    t1, y1, x1 = t[i], y[i], x[i]
    t2, y2, x2 = t[i + 1], y[i + 1], x[i + 1]
    # линейная интерполяция по y==0
    if y2 == y1:
        tf = t2
        xf = x2
    else:
        alpha = -y1 / (y2 - y1)
        tf = t1 + alpha * (t2 - t1)
        xf = x1 + alpha * (x2 - x1)
    return tf, xf


# -------------------------
# Функция запуска одного опыта
# -------------------------
def run_experiment(v0, angle_deg, model, b_coef, c_coef, t_max):
    angle = np.deg2rad(angle_deg)
    v0x = v0 * np.cos(angle)
    v0y = v0 * np.sin(angle)
    state0 = [0.0, 0.0, v0x, v0y]
    t_span = (0.0, t_max)

    results = {}

    if model in ('none', 'both'):
        sol_none = solve_ivp(rhs_none, t_span, state0, max_step=0.05, rtol=rtol, atol=atol, dense_output=True)
        land_none = find_landing_time_and_x(sol_none)
        results['none'] = {'sol': sol_none, 'landing': land_none}

    if model in ('linear', 'both'):
        sol_lin = solve_ivp(lambda t, y: rhs_linear(t, y, b_coef), t_span, state0, max_step=0.05, rtol=rtol, atol=atol,
                            dense_output=True)
        land_lin = find_landing_time_and_x(sol_lin)
        results['linear'] = {'sol': sol_lin, 'landing': land_lin}

    if model in ('quadratic', 'both'):
        sol_quad = solve_ivp(lambda t, y: rhs_quadratic(t, y, c_coef), t_span, state0, max_step=0.02, rtol=rtol,
                             atol=atol, dense_output=True)
        land_quad = find_landing_time_and_x(sol_quad)
        results['quadratic'] = {'sol': sol_quad, 'landing': land_quad}

    return results


# -------------------------
# Визуализация и прогон серии
# -------------------------
def plot_results(results_dict, title_suffix=''):
    plt.figure(figsize=(14, 8))
    all_x = []
    all_y = []

    for key, val in results_dict.items():
        sol = val['sol']
        x = sol.y[0]
        y = sol.y[1]
        label = key
        plt.plot(x, y, label=label, linewidth=2)
        all_x.extend(x)
        all_y.extend(y)

        landing = val['landing']
        if landing is not None:
            tf, xf = landing
            plt.plot([xf], [0.0], 'o')
            plt.annotate(
                f'{key}\n x={xf:.2f} m, t={tf:.2f} s',
                xy=(xf, 0), xycoords='data',
                xytext=(20, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7)
            )

    # Автоматический масштаб по данным
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = 0, max(all_y)  # y_min = 0 (земля), y_max — макс высота

    # Добавляем небольшой запас (например, 10% сверху и справа)
    padding_x = 0.1 * (x_max - x_min)
    padding_y = 0.1 * y_max

    plt.xlim(x_min - padding_x, x_max + padding_x)
    plt.ylim(-0.1, y_max + padding_y)

    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title('Траектории (без сопротивления / линейное / квадратичное) ' + title_suffix)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, 1000)  # от 0 до 1000 метров
    plt.xticks(np.arange(0, x_max + 50, 50))
    plt.yticks(np.arange(0, y_max + 20, 20))
    plt.show()


# -------------------------
# Главная логика: серия запусков
# -------------------------
def main():
    global series
    # если series пуст, взять один опыт из конфигурации
    if not series:
        run_list = [{'v0': v0, 'angle_deg': angle_deg, 'label': 'один'}]
    else:
        run_list = series

    for exp in run_list:
        v0_e = exp.get('v0', v0)
        angle_e = exp.get('angle_deg', angle_deg)
        label = exp.get('label', '')

        print(f"\n=== Эксперимент: {label} v0={v0_e} m/s, angle={angle_e} deg ===")

        # аналитика без сопротивления
        an = analytic_no_drag(v0_e, np.deg2rad(angle_e))
        print("Аналитика (без сопротивления):")
        print(f"  Время полёта: {an['t_fall']:.4f} s, Дальность: {an['range']:.4f} m, Макс высота: {an['h_max']:.4f} m")

        # аналитика линейная (параметры)
        an_lin = analytic_linear(v0_e, np.deg2rad(angle_e), b)
        print(f"  Линейная модель: k = b/m = {an_lin['k']:.6f} 1/s")

        results = run_experiment(v0_e, angle_e, model, b, c, t_max)

        # печать результатов
        for key, val in results.items():
            landing = val['landing']
            if landing is None:
                print(f"  {key}: не упал в пределах t_max={t_max}s")
            else:
                tf, xf = landing
                sol = val['sol']
                # максимальная высота (по массиву)
                ymax = np.max(sol.y[1])
                imax = np.argmax(sol.y[1])
                vx_at_max = sol.y[2, imax]
                vy_at_max = sol.y[3, imax]
                print(f"  {key}: t_fall = {tf:.4f} s, range = {xf:.4f} m, y_max = {ymax:.4f} m")

        plot_results(results, title_suffix=f" ({label})")


if __name__ == "__main__":
    main()
