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

T = np.linspace(-10, 40, 1001)
dT = np.linspace(-2, 2, 801)
knob = np.linspace(-90, 90, 1001)

Tv_cold2 = trimf(T, [-10, -10, 5])
T_cold   = trimf(T, [0, 8, 16])
T_norm   = trimf(T, [18, 21, 24])
T_warm   = trimf(T, [22, 26, 30])
Tv_warm2 = trimf(T, [28, 40, 40])

dT_neg  = trimf(dT, [-2, -2, 0])
dT_zero = trimf(dT, [-0.2, 0, 0.2])
dT_pos  = trimf(dT, [0, 2, 2])

K_large_left  = trapmf(knob, [-90, -90, -75, -60])
K_small_left  = trimf(knob, [-40, -20, 0])
K_zero        = trimf(knob, [-5, 0, 5])
K_small_right = trimf(knob, [0, 20, 40])
K_large_right = trapmf(knob, [60, 75, 90, 90])

def infer_ac(temp_val, dtemp_val):
    muT = {
        'vwarm': np.interp(temp_val, T, Tv_warm2),
        'warm':  np.interp(temp_val, T, T_warm),
        'norm':  np.interp(temp_val, T, T_norm),
        'cold':  np.interp(temp_val, T, T_cold),
        'vcold': np.interp(temp_val, T, Tv_cold2),
    }
    muD = {
        'neg':  np.interp(dtemp_val, dT, dT_neg),
        'zero': np.interp(dtemp_val, dT, dT_zero),
        'pos':  np.interp(dtemp_val, dT, dT_pos),
    }

    K = np.zeros_like(knob)
    def add_rule(alpha, out_set):
        nonlocal K
        K = np.maximum(K, apply_implication(alpha, out_set))

    add_rule(min(muT['vwarm'], muD['pos']),  K_large_left)
    add_rule(min(muT['vwarm'], muD['neg']),  K_small_left)
    add_rule(min(muT['warm'],  muD['pos']),  K_large_left)
    add_rule(min(muT['warm'],  muD['neg']),  K_zero)
    add_rule(min(muT['vcold'], muD['neg']),  K_large_right)
    add_rule(min(muT['vcold'], muD['pos']),  K_small_right)
    add_rule(min(muT['cold'],  muD['neg']),  K_large_right)
    add_rule(min(muT['cold'],  muD['pos']),  K_zero)
    add_rule(min(muT['vwarm'], muD['zero']), K_large_left)
    add_rule(min(muT['warm'],  muD['zero']), K_small_left)
    add_rule(min(muT['vcold'], muD['zero']), K_large_right)
    add_rule(min(muT['cold'],  muD['zero']), K_small_right)
    add_rule(min(muT['norm'],  muD['pos']),  K_small_left)
    add_rule(min(muT['norm'],  muD['neg']),  K_small_right)
    add_rule(min(muT['norm'],  muD['zero']), K_zero)

    angle_out = centroid(knob, K)
    return angle_out, K

def plot_surface():
    Tg, dTg = np.meshgrid(np.linspace(-5,35,41), np.linspace(-1.5,1.5,41))
    vectorized_infer = np.vectorize(lambda t, d: infer_ac(t, d)[0])
    Ksurf = vectorized_infer(Tg, dTg)

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tg, dTg, Ksurf, cmap='viridis', edgecolor='none', alpha=0.95)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('dT (°C/unit)')
    ax.set_zlabel('Knob angle (°)')
    ax.set_title('AC controller surface')
    plt.tight_layout()
    plt.savefig('task2_ac_surface.png', dpi=160)
    plt.show()

if __name__ == '__main__':
    for (tv, dv) in [(30, 0.0), (30, 0.8), (20, 0.0), (12, -0.6), (28, -0.4)]:
        k, _ = infer_ac(tv, dv)
        print(f'T={tv}°C, dT={dv:+.2f} → knob={k:.1f}°')

    plot_surface()