import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --------------------------
# Параметры
# --------------------------
mass = 1.0
radius = 0.05
wall_left, wall_right = -1.0, 1.0
wall_bottom, wall_top = -1.0, 1.0
k_hooke = 1e5  # Жесткость
dt = 1e-4
t_max = 3.0
steps = int(t_max / dt)
t_eval = np.linspace(0, t_max, steps)

# --------------------------
# Начальные условия
# --------------------------
speed1 = 1000.0  # скорость 1-го шара
angle1_deg = 25.0  # угол броска 1-го шара
speed2 = 0.0  # второй изначально покоится
angle2_deg = 0.0  # угол для второго

vx1_0 = speed1 * np.cos(np.radians(angle1_deg))
vy1_0 = speed1 * np.sin(np.radians(angle1_deg))
vx2_0 = speed2 * np.cos(np.radians(angle2_deg))
vy2_0 = speed2 * np.sin(np.radians(angle2_deg))

x1_0, y1_0 = 0.0, 0.0
x2_0, y2_0 = 0.3, 0.0


# --------------------------
# Силы по закону Гука
# --------------------------
def hooke_force_ball_wall(x, y):
    fx = fy = 0.0
    if x - radius < wall_left:
        delta = wall_left - (x - radius)
        fx += k_hooke * delta
    if x + radius > wall_right:
        delta = (x + radius) - wall_right
        fx -= k_hooke * delta
    if y - radius < wall_bottom:
        delta = wall_bottom - (y - radius)
        fy += k_hooke * delta
    if y + radius > wall_top:
        delta = (y + radius) - wall_top
        fy -= k_hooke * delta
    return fx, fy


def hooke_force_ball_ball(x1, y1, x2, y2):
    dx, dy = x1 - x2, y1 - y2
    distance = np.hypot(dx, dy)
    if distance >= 2 * radius or distance == 0:
        return 0.0, 0.0, 0.0, 0.0
    overlap = 2 * radius - distance
    nx, ny = dx / distance, dy / distance
    F = k_hooke * overlap
    return F * nx, F * ny, -F * nx, -F * ny


# --------------------------
# Аналитическая модель (идеально упругие шары)
# --------------------------
def simulate_analytical_two_balls(x1_0, y1_0, vx1_0, vy1_0,
                                  x2_0, y2_0, vx2_0, vy2_0,
                                  t_eval):
    x1, y1, vx1, vy1 = x1_0, y1_0, vx1_0, vy1_0
    x2, y2, vx2, vy2 = x2_0, y2_0, vx2_0, vy2_0
    traj1, traj2 = [], []
    t_prev = 0.0

    for t in t_eval:
        dt_step = t - t_prev
        x1 += vx1 * dt_step
        y1 += vy1 * dt_step
        x2 += vx2 * dt_step
        y2 += vy2 * dt_step

        # столкновение шаров
        dx, dy = x1 - x2, y1 - y2
        distance = np.hypot(dx, dy)
        if distance < 2 * radius:
            nx, ny = dx / distance, dy / distance
            v_rel_n = (vx1 - vx2) * nx + (vy1 - vy2) * ny
            if v_rel_n < 0:  # только при сближении
                vx1 -= v_rel_n * nx
                vy1 -= v_rel_n * ny
                vx2 += v_rel_n * nx
                vy2 += v_rel_n * ny
                overlap = 2 * radius - distance
                x1 += nx * overlap / 2
                y1 += ny * overlap / 2
                x2 -= nx * overlap / 2
                y2 -= ny * overlap / 2

        # отражение от стен
        def reflect(x, y, vx, vy):
            if x - radius < wall_left:
                x = wall_left + radius
                vx = abs(vx)
            elif x + radius > wall_right:
                x = wall_right - radius
                vx = -abs(vx)
            if y - radius < wall_bottom:
                y = wall_bottom + radius
                vy = abs(vy)
            elif y + radius > wall_top:
                y = wall_top - radius
                vy = -abs(vy)
            return x, y, vx, vy

        x1, y1, vx1, vy1 = reflect(x1, y1, vx1, vy1)
        x2, y2, vx2, vy2 = reflect(x2, y2, vx2, vy2)

        traj1.append([x1, y1])
        traj2.append([x2, y2])
        t_prev = t
    return np.array(traj1), np.array(traj2)


analytical_traj1, analytical_traj2 = simulate_analytical_two_balls(
    x1_0, y1_0, vx1_0, vy1_0, x2_0, y2_0, vx2_0, vy2_0, t_eval)

# --------------------------
# Численная модель (закон Гука)
# --------------------------
x1, y1, vx1, vy1 = x1_0, y1_0, vx1_0, vy1_0
x2, y2, vx2, vy2 = x2_0, y2_0, vx2_0, vy2_0
numerical_traj1, numerical_traj2 = [], []


def wall_reflect(x, y, vx, vy):
    if x - radius < wall_left:
        x = wall_left + radius
        vx = abs(vx)
    elif x + radius > wall_right:
        x = wall_right - radius
        vx = -abs(vx)
    if y - radius < wall_bottom:
        y = wall_bottom + radius
        vy = abs(vy)
    elif y + radius > wall_top:
        y = wall_top - radius
        vy = -abs(vy)
    return x, y, vx, vy


for _ in range(steps):
    fx1w, fy1w = hooke_force_ball_wall(x1, y1)
    fx2w, fy2w = hooke_force_ball_wall(x2, y2)
    fx1b, fy1b, fx2b, fy2b = hooke_force_ball_ball(x1, y1, x2, y2)

    ax1 = (fx1w + fx1b) / mass
    ay1 = (fy1w + fy1b) / mass
    ax2 = (fx2w + fx2b) / mass
    ay2 = (fy2w + fy2b) / mass

    vx1 += ax1 * dt
    vy1 += ay1 * dt
    vx2 += ax2 * dt
    vy2 += ay2 * dt

    x1 += vx1 * dt
    y1 += vy1 * dt
    x2 += vx2 * dt
    y2 += vy2 * dt

    x1, y1, vx1, vy1 = wall_reflect(x1, y1, vx1, vy1)
    x2, y2, vx2, vy2 = wall_reflect(x2, y2, vx2, vy2)

    numerical_traj1.append([x1, y1])
    numerical_traj2.append([x2, y2])

numerical_traj1 = np.array(numerical_traj1)
numerical_traj2 = np.array(numerical_traj2)

# --------------------------
# Анимация
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, title in zip([ax1, ax2], ["Analytical (Perfect Elastic)", "Numerical (Hooke's Law)"]):
    ax.set_xlim(wall_left - 0.1, wall_right + 0.1)
    ax.set_ylim(wall_bottom - 0.1, wall_top + 0.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    box = plt.Rectangle((wall_left, wall_bottom),
                        wall_right - wall_left,
                        wall_top - wall_bottom,
                        fill=False, edgecolor='black', lw=2)
    ax.add_patch(box)

ball1_a, = ax1.plot([], [], 'bo', ms=10)
ball2_a, = ax1.plot([], [], 'go', ms=10)
traj1_a, = ax1.plot([], [], 'b-', lw=1, alpha=0.7)
traj2_a, = ax1.plot([], [], 'g-', lw=1, alpha=0.7)

ball1_n, = ax2.plot([], [], 'ro', ms=10)
ball2_n, = ax2.plot([], [], 'mo', ms=10)
traj1_n, = ax2.plot([], [], 'r-', lw=1, alpha=0.7)
traj2_n, = ax2.plot([], [], 'm-', lw=1, alpha=0.7)

traj1_a_x, traj1_a_y, traj2_a_x, traj2_a_y = [], [], [], []
traj1_n_x, traj1_n_y, traj2_n_x, traj2_n_y = [], [], [], []


def update(i):
    if i >= len(analytical_traj1):
        return []

    x1a, y1a = analytical_traj1[i]
    x2a, y2a = analytical_traj2[i]
    ball1_a.set_data([x1a], [y1a])
    ball2_a.set_data([x2a], [y2a])
    traj1_a_x.append(x1a)
    traj1_a_y.append(y1a)
    traj2_a_x.append(x2a)
    traj2_a_y.append(y2a)
    traj1_a.set_data(traj1_a_x, traj1_a_y)
    traj2_a.set_data(traj2_a_x, traj2_a_y)

    x1n, y1n = numerical_traj1[i]
    x2n, y2n = numerical_traj2[i]
    ball1_n.set_data([x1n], [y1n])
    ball2_n.set_data([x2n], [y2n])
    traj1_n_x.append(x1n)
    traj1_n_y.append(y1n)
    traj2_n_x.append(x2n)
    traj2_n_y.append(y2n)
    traj1_n.set_data(traj1_n_x, traj1_n_y)
    traj2_n.set_data(traj2_n_x, traj2_n_y)

    return ball1_a, ball2_a, traj1_a, traj2_a, ball1_n, ball2_n, traj1_n, traj2_n


ani = FuncAnimation(fig, update, frames=len(analytical_traj1) // 10,
                    interval=dt * 1000 * 10, blit=False, repeat=False)
plt.tight_layout()
plt.show()
