import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --------------------------
# Параметры (с нормальными скоростями)
# --------------------------
m1 = 1.0
m2 = 1.0
radius = 0.05
wall_left, wall_right = -1.0, 1.0
wall_bottom, wall_top = -1.0, 1.0
k_hooke = 5e5  # жёсткость для упругого столкновения
c_damp = 100.0  # демпфирование
dt = 1e-4  # шаг по времени
t_max = 1.5  # общее время симуляции
steps = int(t_max / dt)
t_eval = np.linspace(0, t_max, steps)

# --------------------------
# Начальные условия (НОРМАЛЬНЫЕ СКОРОСТИ)
# --------------------------
speed1 = 1
angle1_deg = 0
speed2 = 0.0
angle2_deg = 0.0

vx1_0 = speed1 * np.cos(np.radians(angle1_deg))
vy1_0 = speed1 * np.sin(np.radians(angle1_deg))
vx2_0 = speed2 * np.cos(np.radians(angle2_deg))
vy2_0 = speed2 * np.sin(np.radians(angle2_deg))

# Начальные позиции - обеспечим столкновение
x1_0, y1_0 = -0.2, 0.0
x2_0, y2_0 = 0.0, 0.0

print(f"Начальные скорости:")
print(f"Шар 1: vx={vx1_0:.2f}, vy={vy1_0:.2f}, |v|={np.hypot(vx1_0, vy1_0):.2f}")
print(f"Шар 2: vx={vx2_0:.2f}, vy={vy2_0:.2f}")


# --------------------------
# Функции для расчета сохранения
# --------------------------
def calculate_energy(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """Вычисляет кинетическую энергию системы"""
    kinetic1 = 0.5 * m1 * (vx1**2 + vy1**2)
    kinetic2 = 0.5 * m2 * (vx2**2 + vy2**2)
    return kinetic1 + kinetic2


def calculate_momentum(vx1, vy1, vx2, vy2):
    """Вычисляет импульс системы"""
    px = m1 * vx1 + m2 * vx2
    py = m1 * vy1 + m2 * vy2
    return px, py


def calculate_angular_momentum(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """Вычисляет угловой момент относительно центра системы"""
    # Центр масс
    x_cm = (m1 * x1 + m2 * x2) / (m1 + m2)
    y_cm = (m1 * y1 + m2 * y2) / (m1 + m2)

    # Относительные координаты
    x1_rel = x1 - x_cm
    y1_rel = y1 - y_cm
    x2_rel = x2 - x_cm
    y2_rel = y2 - y_cm

    # Угловой момент
    L1 = m1 * (x1_rel * vy1 - y1_rel * vx1)
    L2 = m2 * (x2_rel * vy2 - y2_rel * vx2)
    return L1 + L2


# --------------------------
# Силы по закону Гука (оптимизированы для высоких скоростей)
# --------------------------
def hooke_force_ball_wall(x, y, vx, vy):
    fx = fy = 0.0
    damping = 50.0  # демпфирование для стен

    if x - radius < wall_left:
        overlap = wall_left - (x - radius)
        fx += k_hooke * overlap
        fx -= damping * vx  # демпфирование
    if x + radius > wall_right:
        overlap = (x + radius) - wall_right
        fx -= k_hooke * overlap
        fx -= damping * vx
    if y - radius < wall_bottom:
        overlap = wall_bottom - (y - radius)
        fy += k_hooke * overlap
        fy -= damping * vy
    if y + radius > wall_top:
        overlap = (y + radius) - wall_top
        fy -= k_hooke * overlap
        fy -= damping * vy
    return fx, fy


def hooke_force_ball_ball(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    dx, dy = x1 - x2, y1 - y2
    distance_sq = dx * dx + dy * dy

    if distance_sq <= 0.0 or distance_sq >= (2 * radius) ** 2:
        return 0.0, 0.0, 0.0, 0.0

    distance = np.sqrt(distance_sq)
    overlap = 2 * radius - distance

    if overlap <= 0:
        return 0.0, 0.0, 0.0, 0.0

    nx, ny = dx / distance, dy / distance

    # Относительная скорость
    v_rel_n = (vx1 - vx2) * nx + (vy1 - vy2) * ny

    # Сила Гука + демпфирование
    F_normal = k_hooke * overlap - c_damp * v_rel_n

    # Ограничим силу снизу чтобы избежать численной нестабильности
    F_normal = max(F_normal, 0)

    fx1 = F_normal * nx
    fy1 = F_normal * ny
    fx2 = -fx1
    fy2 = -fy1

    return fx1, fy1, fx2, fy2


# --------------------------
# Аналитическая модель
# --------------------------
def simulate_analytical_two_balls(
    x1_0,
    y1_0,
    vx1_0,
    vy1_0,
    x2_0,
    y2_0,
    vx2_0,
    vy2_0,
    t_eval,
    m1=1.0,
    m2=1.0,
    radius=0.05,
):
    x1, y1, vx1, vy1 = x1_0, y1_0, vx1_0, vy1_0
    x2, y2, vx2, vy2 = x2_0, y2_0, vx2_0, vy2_0
    traj1, traj2 = [], []
    energy_history = []
    momentum_history = []
    angular_momentum_history = []
    t_prev = 0.0
    collision_processed = False

    # Начальные значения
    initial_energy = calculate_energy(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    initial_px, initial_py = calculate_momentum(vx1, vy1, vx2, vy2)
    initial_angular = calculate_angular_momentum(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

    for t in t_eval:
        dt_step = t - t_prev

        # Движение
        x1 += vx1 * dt_step
        y1 += vy1 * dt_step
        x2 += vx2 * dt_step
        y2 += vy2 * dt_step

        # Столкновение шаров
        dx, dy = x1 - x2, y1 - y2
        distance = np.hypot(dx, dy)

        if 0.0 < distance < 2 * radius and not collision_processed:
            nx, ny = dx / distance, dy / distance
            v_rel_n = (vx1 - vx2) * nx + (vy1 - vy2) * ny

            if v_rel_n < 0:  # только при сближении
                e = 1.0  # абсолютно упругое
                j = -(1 + e) * v_rel_n / (1.0 / m1 + 1.0 / m2)

                vx1 += (j / m1) * nx
                vy1 += (j / m1) * ny
                vx2 -= (j / m2) * nx
                vy2 -= (j / m2) * ny

                # Коррекция позиций
                overlap = 2 * radius - distance
                x1 += nx * (overlap * 0.5)
                y1 += ny * (overlap * 0.5)
                x2 -= nx * (overlap * 0.5)
                y2 -= ny * (overlap * 0.5)

                print(f"Аналитическое столкновение в t={t:.3f}")
                collision_processed = True

        # Отражение от стен
        if x1 - radius < wall_left:
            x1 = wall_left + radius
            vx1 = abs(vx1)
        elif x1 + radius > wall_right:
            x1 = wall_right - radius
            vx1 = -abs(vx1)

        if y1 - radius < wall_bottom:
            y1 = wall_bottom + radius
            vy1 = abs(vy1)
        elif y1 + radius > wall_top:
            y1 = wall_top - radius
            vy1 = -abs(vy1)

        if x2 - radius < wall_left:
            x2 = wall_left + radius
            vx2 = abs(vx2)
        elif x2 + radius > wall_right:
            x2 = wall_right - radius
            vx2 = -abs(vx2)

        if y2 - radius < wall_bottom:
            y2 = wall_bottom + radius
            vy2 = abs(vy2)
        elif y2 + radius > wall_top:
            y2 = wall_top - radius
            vy2 = -abs(vy2)

        traj1.append([x1, y1])
        traj2.append([x2, y2])

        # Сохранение величин
        current_energy = calculate_energy(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
        current_px, current_py = calculate_momentum(vx1, vy1, vx2, vy2)
        current_angular = calculate_angular_momentum(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

        energy_history.append(current_energy)
        momentum_history.append((current_px, current_py))
        angular_momentum_history.append(current_angular)

        t_prev = t

    conservation_data = {
        "energy": energy_history,
        "momentum": momentum_history,
        "angular_momentum": angular_momentum_history,
        "initial_energy": initial_energy,
        "initial_momentum": (initial_px, initial_py),
        "initial_angular": initial_angular,
    }

    return np.array(traj1), np.array(traj2), conservation_data


print("Запуск аналитической симуляции...")
analytical_traj1, analytical_traj2, analytical_conservation = (
    simulate_analytical_two_balls(
        x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0, t_eval, m1, m2, radius
    )
)

# --------------------------
# Численная модель
# --------------------------
print("Запуск численной симуляции...")
x1, y1, vx1, vy1 = x1_0, y1_0, vx1_0, vy1_0
x2, y2, vx2, vy2 = x2_0, y2_0, vx2_0, vy2_0
numerical_traj1, numerical_traj2 = [], []
numerical_energy_history = []
numerical_momentum_history = []
numerical_angular_momentum_history = []

# Начальные значения
initial_energy = calculate_energy(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
initial_px, initial_py = calculate_momentum(vx1, vy1, vx2, vy2)
initial_angular = calculate_angular_momentum(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

for i in range(steps):
    # Силы
    fx1w, fy1w = hooke_force_ball_wall(x1, y1, vx1, vy1)
    fx2w, fy2w = hooke_force_ball_wall(x2, y2, vx2, vy2)
    fx1b, fy1b, fx2b, fy2b = hooke_force_ball_ball(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

    # Ускорения
    ax1 = (fx1w + fx1b) / m1
    ay1 = (fy1w + fy1b) / m1
    ax2 = (fx2w + fx2b) / m2
    ay2 = (fy2w + fy2b) / m2

    # Интегрирование (Верле или простой Эйлер)
    vx1 += ax1 * dt
    vy1 += ay1 * dt
    vx2 += ax2 * dt
    vy2 += ay2 * dt

    x1 += vx1 * dt
    y1 += vy1 * dt
    x2 += vx2 * dt
    y2 += vy2 * dt

    numerical_traj1.append([x1, y1])
    numerical_traj2.append([x2, y2])

    # Сохранение величин
    current_energy = calculate_energy(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    current_px, current_py = calculate_momentum(vx1, vy1, vx2, vy2)
    current_angular = calculate_angular_momentum(x1, y1, vx1, vy1, x2, y2, vx2, vy2)

    numerical_energy_history.append(current_energy)
    numerical_momentum_history.append((current_px, current_py))
    numerical_angular_momentum_history.append(current_angular)

numerical_traj1 = np.array(numerical_traj1)
numerical_traj2 = np.array(numerical_traj2)

numerical_conservation = {
    "energy": numerical_energy_history,
    "momentum": numerical_momentum_history,
    "angular_momentum": numerical_angular_momentum_history,
    "initial_energy": initial_energy,
    "initial_momentum": (initial_px, initial_py),
    "initial_angular": initial_angular,
}

# --------------------------
# Анимация
# --------------------------
fig, (ax1p, ax2p) = plt.subplots(1, 2, figsize=(14, 6))

for ax, title in zip(
    [ax1p, ax2p], ["Analytical (Perfect Elastic)", "Numerical (Hooke's Law)"]
):
    ax.set_xlim(wall_left - 0.1, wall_right + 0.1)
    ax.set_ylim(wall_bottom - 0.1, wall_top + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    box = plt.Rectangle(
        (wall_left, wall_bottom),
        wall_right - wall_left,
        wall_top - wall_bottom,
        fill=False,
        edgecolor="black",
        lw=2,
    )
    ax.add_patch(box)

# Аналитическая модель
(ball1_a,) = ax1p.plot([], [], "bo", ms=12, label="Ball 1")
(ball2_a,) = ax1p.plot([], [], "ro", ms=12, label="Ball 2")
(traj1_a,) = ax1p.plot([], [], "b-", lw=1, alpha=0.6)
(traj2_a,) = ax1p.plot([], [], "r-", lw=1, alpha=0.6)
ax1p.legend()

# Численная модель
(ball1_n,) = ax2p.plot([], [], "bo", ms=12, label="Ball 1")
(ball2_n,) = ax2p.plot([], [], "ro", ms=12, label="Ball 2")
(traj1_n,) = ax2p.plot([], [], "b-", lw=1, alpha=0.6)
(traj2_n,) = ax2p.plot([], [], "r-", lw=1, alpha=0.6)
ax2p.legend()

# Данные для траекторий
traj_data = {"a1": [[], []], "a2": [[], []], "n1": [[], []], "n2": [[], []]}

# Ограничим количество кадров для плавности
n_frames = 100
step = max(1, len(analytical_traj1) // n_frames)


def update(frame):
    i = frame * step
    if i >= len(analytical_traj1):
        i = len(analytical_traj1) - 1

    # Аналитическая
    x1a, y1a = analytical_traj1[i]
    x2a, y2a = analytical_traj2[i]
    ball1_a.set_data([x1a], [y1a])
    ball2_a.set_data([x2a], [y2a])

    traj_data["a1"][0].append(x1a)
    traj_data["a1"][1].append(y1a)
    traj_data["a2"][0].append(x2a)
    traj_data["a2"][1].append(y2a)

    traj1_a.set_data(traj_data["a1"][0], traj_data["a1"][1])
    traj2_a.set_data(traj_data["a2"][0], traj_data["a2"][1])

    # Численная
    x1n, y1n = numerical_traj1[i]
    x2n, y2n = numerical_traj2[i]
    ball1_n.set_data([x1n], [y1n])
    ball2_n.set_data([x2n], [y2n])

    traj_data["n1"][0].append(x1n)
    traj_data["n1"][1].append(y1n)
    traj_data["n2"][0].append(x2n)
    traj_data["n2"][1].append(y2n)

    traj1_n.set_data(traj_data["n1"][0], traj_data["n1"][1])
    traj2_n.set_data(traj_data["n2"][0], traj_data["n2"][1])

    return ball1_a, ball2_a, traj1_a, traj2_a, ball1_n, ball2_n, traj1_n, traj2_n


print("Создание анимации...")
ani = FuncAnimation(
    fig,
    update,
    frames=min(n_frames, len(analytical_traj1) // step),
    interval=20,
    blit=False,
    repeat=True,
)

plt.tight_layout()
plt.show()

# --------------------------
# Анализ сохранения законов
# --------------------------
print("\n" + "=" * 50)
print("ПРОВЕРКА ЗАКОНОВ СОХРАНЕНИЯ")
print("=" * 50)

# Аналитическая модель
analytical_energy = analytical_conservation["energy"]
analytical_momentum = analytical_conservation["momentum"]
analytical_angular = analytical_conservation["angular_momentum"]

# Численная модель
numerical_energy = numerical_conservation["energy"]
numerical_momentum = numerical_conservation["momentum"]
numerical_angular = numerical_conservation["angular_momentum"]

# Анализ энергии
print("\n--- СОХРАНЕНИЕ ЭНЕРГИИ ---")
print(f"Начальная энергия: {analytical_conservation['initial_energy']:.6f}")

final_energy_analytical = analytical_energy[-1]
final_energy_numerical = numerical_energy[-1]

energy_change_analytical = (
    abs(final_energy_analytical - analytical_conservation["initial_energy"])
    / analytical_conservation["initial_energy"]
    * 100
)
energy_change_numerical = (
    abs(final_energy_numerical - numerical_conservation["initial_energy"])
    / numerical_conservation["initial_energy"]
    * 100
)

print(f"Аналитическая модель:")
print(f"  Конечная энергия: {final_energy_analytical:.6f}")
print(f"  Изменение энергии: {energy_change_analytical:.4f}%")

print(f"Численная модель:")
print(f"  Конечная энергия: {final_energy_numerical:.6f}")
print(f"  Изменение энергии: {energy_change_numerical:.4f}%")

# Анализ импульса
print("\n--- СОХРАНЕНИЕ ИМПУЛЬСА ---")
initial_px, initial_py = analytical_conservation["initial_momentum"]
print(f"Начальный импульс: Px={initial_px:.6f}, Py={initial_py:.6f}")

final_px_analytical, final_py_analytical = analytical_momentum[-1]
final_px_numerical, final_py_numerical = numerical_momentum[-1]

momentum_change_analytical = np.sqrt(
    (final_px_analytical - initial_px) ** 2 + (final_py_analytical - initial_py) ** 2
)
momentum_change_numerical = np.sqrt(
    (final_px_numerical - initial_px) ** 2 + (final_py_numerical - initial_py) ** 2
)

print(f"Аналитическая модель:")
print(f"  Конечный импульс: Px={final_px_analytical:.6f}, Py={final_py_analytical:.6f}")
print(f"  Изменение импульса: {momentum_change_analytical:.6f}")

print(f"Численная модель:")
print(f"  Конечный импульс: Px={final_px_numerical:.6f}, Py={final_py_numerical:.6f}")
print(f"  Изменение импульса: {momentum_change_numerical:.6f}")

# Анализ углового момента
print("\n--- СОХРАНЕНИЕ УГЛОВОГО МОМЕНТА ---")
initial_angular = analytical_conservation["initial_angular"]
print(f"Начальный угловой момент: {initial_angular:.6f}")

final_angular_analytical = analytical_angular[-1]
final_angular_numerical = numerical_angular[-1]

angular_change_analytical = (
    abs(final_angular_analytical - initial_angular) / abs(initial_angular) * 100
)
angular_change_numerical = (
    abs(final_angular_numerical - initial_angular) / abs(initial_angular) * 100
)

print(f"Аналитическая модель:")
print(f"  Конечный угловой момент: {final_angular_analytical:.6f}")
print(f"  Изменение: {angular_change_analytical:.4f}%")

print(f"Численная модель:")
print(f"  Конечный угловой момент: {final_angular_numerical:.6f}")
print(f"  Изменение: {angular_change_numerical:.4f}%")

# Графики сохранения
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle("Проверка законов сохранения", fontsize=16)

# Энергия
axes[0, 0].plot(
    t_eval[: len(analytical_energy)], analytical_energy, "b-", label="Аналитическая"
)
axes[0, 0].plot(
    t_eval[: len(numerical_energy)], numerical_energy, "r-", label="Численная"
)
axes[0, 0].axhline(
    y=analytical_conservation["initial_energy"],
    color="k",
    linestyle="--",
    alpha=0.7,
    label="Начальная",
)
axes[0, 0].set_xlabel("Время (с)")
axes[0, 0].set_ylabel("Энергия")
axes[0, 0].set_title("Кинетическая энергия")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Импульс X
analytical_px = [p[0] for p in analytical_momentum]
numerical_px = [p[0] for p in numerical_momentum]
axes[0, 1].plot(
    t_eval[: len(analytical_px)], analytical_px, "b-", label="Аналитическая"
)
axes[0, 1].plot(t_eval[: len(numerical_px)], numerical_px, "r-", label="Численная")
axes[0, 1].axhline(
    y=initial_px, color="k", linestyle="--", alpha=0.7, label="Начальный"
)
axes[0, 1].set_xlabel("Время (с)")
axes[0, 1].set_ylabel("Px")
axes[0, 1].set_title("Импульс по X")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Импульс Y
analytical_py = [p[1] for p in analytical_momentum]
numerical_py = [p[1] for p in numerical_momentum]
axes[0, 2].plot(
    t_eval[: len(analytical_py)], analytical_py, "b-", label="Аналитическая"
)
axes[0, 2].plot(t_eval[: len(numerical_py)], numerical_py, "r-", label="Численная")
axes[0, 2].axhline(
    y=initial_py, color="k", linestyle="--", alpha=0.7, label="Начальный"
)
axes[0, 2].set_xlabel("Время (с)")
axes[0, 2].set_ylabel("Py")
axes[0, 2].set_title("Импульс по Y")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Относительное изменение энергии
rel_energy_analytical = (
    np.abs(np.array(analytical_energy) - analytical_conservation["initial_energy"])
    / analytical_conservation["initial_energy"]
    * 100
)
rel_energy_numerical = (
    np.abs(np.array(numerical_energy) - numerical_conservation["initial_energy"])
    / numerical_conservation["initial_energy"]
    * 100
)

axes[1, 0].plot(
    t_eval[: len(rel_energy_analytical)],
    rel_energy_analytical,
    "b-",
    label="Аналитическая",
)
axes[1, 0].plot(
    t_eval[: len(rel_energy_numerical)], rel_energy_numerical, "r-", label="Численная"
)
axes[1, 0].set_xlabel("Время (с)")
axes[1, 0].set_ylabel("Изменение (%)")
axes[1, 0].set_title("Относительное изменение энергии")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale("log")

# Угловой момент
axes[1, 1].plot(
    t_eval[: len(analytical_angular)], analytical_angular, "b-", label="Аналитическая"
)
axes[1, 1].plot(
    t_eval[: len(numerical_angular)], numerical_angular, "r-", label="Численная"
)
axes[1, 1].axhline(
    y=initial_angular, color="k", linestyle="--", alpha=0.7, label="Начальный"
)
axes[1, 1].set_xlabel("Время (с)")
axes[1, 1].set_ylabel("Угловой момент")
axes[1, 1].set_title("Угловой момент")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Сводка
models = ["Аналитическая", "Численная"]
energy_changes = [energy_change_analytical, energy_change_numerical]
bars = axes[1, 2].bar(models, energy_changes, color=["blue", "red"], alpha=0.7)
axes[1, 2].set_ylabel("Изменение энергии (%)")
axes[1, 2].set_title("Сводка по сохранению энергии")
axes[1, 2].grid(True, alpha=0.3)

# Добавим значения на столбцы
for bar, value in zip(bars, energy_changes):
    axes[1, 2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.4f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()
