import pygame
import math
import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------
# Входные данные
# -----------------------------
ANGLE_DEG = 15
BALL_RADIUS = 20
MU_STATIC = 0.15
MU_KINETIC = 0.15
g = 980
MASS = 1.0
I = (2 / 5) * MASS * BALL_RADIUS**2

INIT_X, INIT_Y = 150, 300

# -----------------------------
# Pygame init
# -----------------------------
pygame.init()
WIDTH, HEIGHT = 1800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 24)

WHITE = (255, 255, 255)
RED = (200, 50, 50)
GRAY = (180, 180, 180)
BLUE = (50, 100, 200)
BLACK = (0, 0, 0)

# -----------------------------
# Физика
# -----------------------------
theta = math.radians(ANGLE_DEG)
cos_t = math.cos(theta)
sin_t = math.sin(theta)

# Точка касания в начальный момент
norm_x = math.sin(theta)
norm_y = math.cos(theta)
contact_x = INIT_X - BALL_RADIUS * norm_x
contact_y = INIT_Y - BALL_RADIUS * norm_y

dir_x = math.cos(theta)
dir_y = -math.sin(theta)

N = MASS * g * cos_t
F_friction_req = (2 / 7) * MASS * g * sin_t
F_friction_max = MU_STATIC * N
print(F_friction_req, F_friction_max)
can_roll_without_slipping = F_friction_req <= F_friction_max


# -----------------------------
# Дифференциальные уравнения
# -----------------------------
def equations(t, y):
    s, v, omega, phi = y
    slip_speed = v - omega * BALL_RADIUS
    # print(abs(slip_speed), abs(2e-2 * v))
    if can_roll_without_slipping and abs(slip_speed) <= abs(2e-2 * v):
        # Чистое качение
        a = (5 / 7) * g * sin_t
        alpha = a / BALL_RADIUS
    else:
        # Скольжение
        direction = -1 if slip_speed > 0 else 1
        F_friction = MU_KINETIC * N * direction
        a = g * sin_t + F_friction / MASS
        alpha = -(F_friction * BALL_RADIUS) / I

    return [v, a, alpha, omega]


# -----------------------------
# Предварительное интегрирование
# -----------------------------
t_span = (0, 10)  # 10 секунд моделирования
y0 = [0.0, 0.0, 0.0, 0.0]  # [s, v_cm, omega, phi]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(
    equations,
    t_span,
    y0,
    t_eval=t_eval,
)

times = sol.t
s_vals = sol.y[0]
v_vals = sol.y[1]
omega_vals = sol.y[2]
rotation_angle_vals = sol.y[3]


def world_to_screen(x, y):
    return (x, HEIGHT - y)


def get_center_from_s(s_val):
    px = contact_x + s_val * dir_x
    py = contact_y + s_val * dir_y
    cx = px + BALL_RADIUS * norm_x
    cy = py + BALL_RADIUS * norm_y
    return cx, cy


start_line, end_line = (
    (contact_x - 4000 * dir_x, contact_y - 4000 * dir_y),
    (contact_x + 4000 * dir_x, contact_y + 4000 * dir_y),
)

# -----------------------------
# Основной цикл визуализации
# -----------------------------
running = True
frame = 0
frame_rate = 50  # кадров в секунду
dt_frame = 1.0 / frame_rate

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                frame = 0  # перезапуск анимации

    if frame >= len(times):
        frame = len(times) - 1

    t = times[frame]
    s = s_vals[frame]
    v_cm = v_vals[frame]
    omega = omega_vals[frame]
    rotation_angle = -rotation_angle_vals[frame]

    # Центр шара
    cx, cy = get_center_from_s(s)

    # Энергии
    delta_h = -s * sin_t
    potential_energy = MASS * g * delta_h
    kinetic_trans = 0.5 * MASS * v_cm**2
    kinetic_rot = 0.5 * I * omega**2
    total_energy = potential_energy + kinetic_trans + kinetic_rot
    angular_momentum = I * omega

    # -----------------------------
    # Визуализация
    # -----------------------------
    screen.fill(WHITE)
    pygame.draw.line(
        screen, GRAY, world_to_screen(*start_line), world_to_screen(*end_line), 6
    )
    pygame.draw.circle(screen, RED, world_to_screen(cx, cy), BALL_RADIUS)

    marker_x = cx + BALL_RADIUS * math.cos(rotation_angle)
    marker_y = cy + BALL_RADIUS * math.sin(rotation_angle)
    pygame.draw.circle(screen, BLUE, world_to_screen(marker_x, marker_y), 4)

    # Текст
    info = [
        f"v = {v_cm:.1f} px/s",
        f"ω = {omega:.2f} rad/s",
        f"Slipping: {'YES' if not (can_roll_without_slipping and abs(v_cm - omega * BALL_RADIUS) <= abs(2e-2 * v_cm)) else 'NO'}",
        f"E_total = {total_energy:.1f}",
        f"Time = {t:.3f} s",
        f"L = {angular_momentum:.2f} (Iω)",
    ]
    for i, txt in enumerate(info):
        surf = FONT.render(txt, True, BLACK)
        screen.blit(surf, (10, 10 + i * 20))

    pygame.display.flip()
    clock.tick(frame_rate)

    frame += 1

pygame.quit()
