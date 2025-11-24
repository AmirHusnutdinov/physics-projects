import pygame
import math

# -----------------------------
# Входные данные
# -----------------------------
ANGLE_DEG = 15  # угол наклона в градусах
BALL_RADIUS = 20  # радиус шара
MU_STATIC = 0.35  # коэффициент статического трения
MU_KINETIC = 0.05  # коэффициент кинетического трения
g = 980  # ускорение (в пикселях/с², чтобы было видно движение)
MASS = 1.0
I = (2 / 5) * MASS * BALL_RADIUS**2  # момент инерции сплошного шара

# Начальное положение ЦЕНТРА шара
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


def compute_incline_line(angle_deg, contact_point, length=4000):
    """Рисует наклонную плоскость длиной 'length', проходящую через contact_point под углом angle_deg к горизонту."""
    angle_rad = math.radians(angle_deg)
    cx = INIT_X - BALL_RADIUS * sin_t
    cy = INIT_Y - BALL_RADIUS * cos_t

    # Вектор вдоль плоскости
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)
    # print(dx, dy)
    # Начало и конец линии
    x1 = cx - length * dx
    y1 = cy - length * dy
    x2 = cx + length * dx
    y2 = cy + length * dy
    return (x1, y1), (x2, y2)


# Вектор вдоль плоскости (единичный)
dir_x = math.cos(theta)
dir_y = -math.sin(theta)

# Перпендикуляр (нормаль вверх от плоскости)
norm_x = math.sin(theta)
norm_y = math.cos(theta)

# Точка касания шара с плоскостью (на поверхности)
contact_x = INIT_X - BALL_RADIUS * norm_x
contact_y = INIT_Y - BALL_RADIUS * norm_y

s = 0
v_cm = 30.0  # скорость вдоль плоскости
omega = 0.0
rotation_angle = 0.0
t = 0.0
dt = 0.01


# -----------------------------
# Вспомогательные функции
# -----------------------------
def world_to_screen(x, y):
    return (x, HEIGHT - y)


def get_center_from_s(s_val):
    """Возвращает (x, y) центра шара по расстоянию s вдоль плоскости от точки касания."""
    # Точка на плоскости на расстоянии s от contact
    px = contact_x + s_val * dir_x
    py = contact_y + s_val * dir_y
    # Центр шара — на нормали на расстоянии R
    cx = px + BALL_RADIUS * norm_x  # norm_x = sinθ
    cy = py + BALL_RADIUS * norm_y  # norm_y = cosθ
    return cx, cy


start_line, end_line = compute_incline_line(ANGLE_DEG, (contact_x, contact_y))

# -----------------------------
# Основной цикл
# -----------------------------
time = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                s = (INIT_X - contact_x) * dir_x + (INIT_Y - contact_y) * dir_y
                v_cm = 0.0
                omega = 0.0
                rotation_angle = 0.0
                t = 0.0
                E_initial = None

    # -----------------------------
    # Физика
    # -----------------------------
    time += dt
    N = MASS * g * cos_t

    F_friction_req = (2 / 7) * MASS * g * sin_t
    F_friction_max = MU_STATIC * N

    can_roll_without_slipping = F_friction_req <= F_friction_max
    if time == dt:
        print(F_friction_req, F_friction_max)
    if can_roll_without_slipping and abs(v_cm) < 1e-6 and abs(omega) < 1e-6:
        slipping = False
    elif can_roll_without_slipping:
        slip_speed = v_cm - omega * BALL_RADIUS
        slipping = abs(slip_speed) > 1e-3
    else:
        slipping = True

    if not slipping:
        a_cm = (5 / 7) * g * sin_t
        alpha = a_cm / BALL_RADIUS
    else:
        slip_speed = v_cm - omega * BALL_RADIUS
        direction = -1 if slip_speed > 0 else 1
        F_friction = MU_KINETIC * N * direction
        a_cm = g * sin_t + (F_friction / MASS)
        alpha = -(F_friction * BALL_RADIUS) / I

    # Обновление
    v_cm += a_cm * dt
    omega += alpha * dt
    s += v_cm * dt
    rotation_angle -= omega * dt
    t += dt

    # Получаем центр шара
    cx, cy = get_center_from_s(s)
    # -----------------------------
    # Энергия и момент импульса
    # -----------------------------
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
    # Рисуем плоскость
    pygame.draw.line(
        screen, GRAY, world_to_screen(*start_line), world_to_screen(*end_line), 6
    )

    # Рисуем шар
    pygame.draw.circle(screen, RED, world_to_screen(cx, cy), BALL_RADIUS)

    # Маркер вращения
    marker_x = cx + BALL_RADIUS * math.cos(rotation_angle)
    marker_y = cy + BALL_RADIUS * math.sin(rotation_angle)
    pygame.draw.circle(screen, BLUE, world_to_screen(marker_x, marker_y), 4)

    # Информация
    info1 = FONT.render(f"v = {v_cm:.1f} px/s", True, BLACK)
    info2 = FONT.render(f"ω = {omega:.2f} rad/s", True, BLACK)
    info3 = FONT.render(f"Slipping: {'YES' if slipping else 'NO'}", True, BLACK)
    info4 = FONT.render(
        f"E_total = {total_energy:.1f}",
        True,
        BLACK,
    )
    info5 = FONT.render(f"Time {time:.3f}", True, BLACK)
    info6 = FONT.render(f"L = {angular_momentum:.2f} (Iω)", True, BLACK)

    screen.blit(info1, (10, 10))
    screen.blit(info2, (10, 30))
    screen.blit(info3, (10, 50))
    screen.blit(info4, (10, 70))
    screen.blit(info5, (10, 90))
    screen.blit(info6, (10, 110))

    pygame.display.flip()
    clock.tick(50)

pygame.quit()
