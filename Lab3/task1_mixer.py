import numpy as np
import matplotlib.pyplot as plt

def trimf(x, abc):
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    idx_left = np.logical_and(a < x, x < b)
    idx_right = np.logical_and(b < x, x < c)
    if b != a:
        y[idx_left] = (x[idx_left] - a) / (b - a)
    y[x == b] = 1.0
    if c != b:
        y[idx_right] = (c - x[idx_right]) / (c - b)
    return np.clip(y, 0, 1)

def trapmf(x, abcd):
    a, b, c, d = abcd
    y = np.zeros_like(x, dtype=float)
    y[np.logical_and(b <= x, x <= c)] = 1.0
    idx_rise = np.logical_and(a < x, x < b)
    idx_fall = np.logical_and(c < x, x < d)
    if b != a:
        y[idx_rise] = (x[idx_rise] - a) / (b - a)
    if d != c:
        y[idx_fall] = (d - x[idx_fall]) / (d - c)
    return np.clip(y, 0, 1)

def centroid(x, mu):
    area = np.trapezoid(mu, x)
    if area == 0:
        return 0.5 * (x.min() + x.max())
    return np.trapezoid(mu * x, x) / area

def apply_implication(alpha, mf):
    return np.minimum(alpha, mf)

# Універсальні універсальні змінні
temp = np.linspace(0, 100, 1001)
press = np.linspace(0, 100, 1001)
angle = np.linspace(-90, 90, 1001)

# Температура
T_cold  = trimf(temp, [0, 0, 20])
T_cool  = trimf(temp, [10, 25, 40])
T_warm  = trimf(temp, [30, 45, 60])
T_nvhot = trimf(temp, [50, 65, 80])
T_hot   = trimf(temp, [70, 100, 100])

# Тиск
P_weak   = trimf(press, [0, 0, 35])
P_nvs    = trimf(press, [25, 50, 75])
P_strong = trimf(press, [60, 100, 100])

# Кути
A_large_left  = trapmf(angle, [-90, -90, -75, -60])
A_medium_left = trimf(angle, [-70, -40, -10])
A_small_left  = trimf(angle, [-25, -10, -2])
A_zero        = trimf(angle, [-5, 0, 5])
A_small_right = trimf(angle, [2, 10, 25])
A_medium_right= trimf(angle, [10, 40, 70])
A_large_right = trapmf(angle, [60, 75, 90, 90])

def infer_mixer(t_val, p_val):
    t_mu = {
        'hot':   np.interp(t_val, temp, T_hot),
        'nvhot': np.interp(t_val, temp, T_nvhot),
        'warm':  np.interp(t_val, temp, T_warm),
        'cool':  np.interp(t_val, temp, T_cool),
        'cold':  np.interp(t_val, temp, T_cold)
    }
    p_mu = {
        'strong': np.interp(p_val, press, P_strong),
        'nvs':    np.interp(p_val, press, P_nvs),
        'weak':   np.interp(p_val, press, P_weak)
    }

    Ah = np.zeros_like(angle)
    Ac = np.zeros_like(angle)

    sets = {
        'LL': A_large_left, 'ML': A_medium_left, 'SL': A_small_left,
        'ZR': A_zero,
        'SR': A_small_right, 'MR': A_medium_right, 'LR': A_large_right
    }
    def S(name): return sets[name]
    def add_rule(alpha, out_hot, out_cold):
        nonlocal Ah, Ac
        Ah = np.maximum(Ah, apply_implication(alpha, out_hot))
        Ac = np.maximum(Ac, apply_implication(alpha, out_cold))

    # Правила
    add_rule(min(t_mu['hot'],   p_mu['strong']), S('ML'), S('MR'))
    add_rule(min(t_mu['hot'],   p_mu['nvs']),    S('ZR'), S('MR'))
    add_rule(min(t_mu['nvhot'], p_mu['strong']), S('SL'), S('ZR'))
    add_rule(min(t_mu['nvhot'], p_mu['weak']),   S('SR'), S('SR'))
    add_rule(min(t_mu['warm'],  p_mu['nvs']),    S('ZR'), S('ZR'))
    add_rule(min(t_mu['cool'],  p_mu['strong']), S('MR'), S('ML'))
    add_rule(min(t_mu['cool'],  p_mu['nvs']),    S('MR'), S('SL'))
    add_rule(min(t_mu['cold'],  p_mu['weak']),   S('LR'), S('ZR'))
    add_rule(min(t_mu['cold'],  p_mu['strong']), S('ML'), S('MR'))
    add_rule(min(t_mu['warm'],  p_mu['strong']), S('SL'), S('SL'))
    add_rule(min(t_mu['warm'],  p_mu['weak']),   S('SR'), S('SR'))

    hot_angle = centroid(angle, Ah)
    cold_angle = centroid(angle, Ac)
    return hot_angle, cold_angle

def plot_surfaces():
    Tg, Pg = np.meshgrid(np.linspace(0,100,41), np.linspace(0,100,41))
    vectorized_infer = np.vectorize(lambda t, p: infer_mixer(t, p))
    Hsurf, Csurf = vectorized_infer(Tg, Pg)

    fig = plt.figure(figsize=(11,4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Tg, Pg, Hsurf, cmap='coolwarm', edgecolor='none', alpha=0.95)
    ax1.set_title('Hot tap angle (°)')
    ax1.set_xlabel('Temperature'); ax1.set_ylabel('Pressure'); ax1.set_zlabel('Angle')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(Tg, Pg, Csurf, cmap='coolwarm', edgecolor='none', alpha=0.95)
    ax2.set_title('Cold tap angle (°)')
    ax2.set_xlabel('Temperature'); ax2.set_ylabel('Pressure'); ax2.set_zlabel('Angle')

    plt.tight_layout()
    plt.savefig('task1_mixer_surfaces.png', dpi=160)
    plt.show()

if __name__ == '__main__':
    ha, ca = infer_mixer(80, 70)
    print(f'Example: T=80, P=70 → hot={ha:.1f}°, cold={ca:.1f}°')
    plot_surfaces()