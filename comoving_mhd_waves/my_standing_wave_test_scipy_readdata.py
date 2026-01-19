import numpy as np
import matplotlib.pyplot as plt
from comoving_mhd_waves import ScipyComovingMagnetosonicWave

# ========================================
# 读取 GAMER 数据
# ========================================
def load_gamer_data(filename):
    """
    读取你的 txt 文件格式
    """
    a_values = []
    delta_rho_values = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # 提取 a_values
    a_start = content.find('a_values = np.array([')
    a_end = content.find('])', a_start) + 2
    a_str = content[a_start:a_end]
    # 执行提取的代码
    exec(a_str, {'np': np}, locals_dict := {})
    a_values = locals_dict. get('a_values')
    
    # 提取 delta_rho_norm_lefthalf
    rho_start = content.find('delta_rho_norm_lefthalf = np.array([')
    rho_end = content.find('])', rho_start) + 2
    rho_str = content[rho_start:rho_end]
    exec(rho_str, {'np': np}, locals_dict := {})
    delta_rho_values = locals_dict.get('delta_rho_norm_lefthalf')
    
    return a_values, delta_rho_values

# 读取数据文件
a_gamer, delta_rho_gamer = load_gamer_data('../deltarho_norm_data_lefthalf-MHM_RP-CFL0.1.txt')

# ========================================
# 解析解参数
# ========================================
k = 2 * np.pi
H0 = 1.0
gamma = 5/3
ai = 1.0/128.0

rho_sim = 1.0
P_sim = 1.0
Bz_sim = 1.0

Vs_sim = np.sqrt(gamma * P_sim / rho_sim)
Va_sim = Bz_sim / np.sqrt(rho_sim)
Vg_sim = np.sqrt(1.5) * H0 / k

A_u_sim = 1j * (1.0e-5 / rho_sim) / Vs_sim / ai
A_rho_sim = 0.0

# 使用 Scipy 数值求解器
sol = ScipyComovingMagnetosonicWave(k, H0, Vs_sim, Va_sim, Vg_sim, gamma, ai, A_u_sim, A_rho_sim)

# ========================================
# 画图
# ========================================
a = np.logspace(np.log10(ai), 0, 300000)

fig, axes = plt. subplots(nrows=2, ncols=2, figsize=(12, 9), sharex=True)

# 密度扰动 - 实部 (主要对比图)
axes[0, 0].semilogx(a, sol.delta_rhoc_over_rhoc(a). real, 'b-', linewidth=2, label='Analytic (Real)')
axes[0, 0].semilogx(a_gamer, delta_rho_gamer, 'ro', markersize=3, alpha=0.7, label='GAMER data')
axes[0, 0].set_title(r'$\mathrm{Re}(\delta \rho_c/\rho_c)$ — Compare Here! ', fontsize=12)
axes[0, 0].set_ylabel(r'$\delta \rho / \rho$')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)

# 密度扰动 - 虚部
axes[0, 1].semilogx(a, sol.delta_rhoc_over_rhoc(a).imag, 'orange', linewidth=2, label='Analytic (Imag)')
axes[0, 1]. semilogx(a_gamer, delta_rho_gamer, 'ro', markersize=3, alpha=0.3, label='GAMER data (for ref)')
axes[0, 1].set_title(r'$\mathrm{Im}(\delta \rho_c/\rho_c)$', fontsize=12)
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3)

# 速度扰动 - 实部
axes[1, 0].semilogx(a, sol.delta_u_over_vs(a).real, 'g-', linewidth=2, label='Analytic (Real)')
axes[1, 0].set_title(r'$\mathrm{Re}(\delta u/V_s)$', fontsize=12)
axes[1, 0].set_xlabel('Scale factor a')
axes[1, 0].set_ylabel(r'$\delta u / V_s$')
axes[1, 0]. legend(loc='best')
axes[1, 0].grid(True, alpha=0.3)

# 速度扰动 - 虚部
axes[1, 1].semilogx(a, sol. delta_u_over_vs(a). imag, 'r-', linewidth=2, label='Analytic (Imag)')
axes[1, 1]. set_title(r'$\mathrm{Im}(\delta u/V_s)$', fontsize=12)
axes[1, 1].set_xlabel('Scale factor a')
axes[1, 1].legend(loc='best')
axes[1, 1].grid(True, alpha=0.3)

# 总标题
plt.suptitle(f'GAMER vs Analytic Solution (γ={gamma:.2f})', fontsize=14)
plt.tight_layout()
plt.savefig('gamer_vs_analytic.png', dpi=150)
plt.show()

# ========================================
# 单独画一张重点对比图
# ========================================
plt.figure(figsize=(10, 6))
plt.semilogx(a, sol.delta_rhoc_over_rhoc(a). real, 'b-', linewidth=2, label='Analytic Solution (Real part)')
plt.semilogx(a_gamer, delta_rho_gamer, 'ro', markersize=4, alpha=0.7, label='GAMER Simulation')
plt. xlabel('Scale factor a', fontsize=12)
plt.ylabel(r'$\delta \rho_c / \rho_c$', fontsize=12)
plt.title('Density Perturbation: GAMER vs Analytic', fontsize=14)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gamer_vs_analytic_density.png', dpi=150)
plt.show()

# ========================================
# 打印信息
# ========================================
print("=" * 50)
print("Parameters:")
print(f"  γ = {gamma}")
print(f"  ai = {ai}")
print(f"  Vs = {Vs_sim:.6f}")
print(f"  Va = {Va_sim:.6f}")
print(f"  Vg = {Vg_sim:.6f}")
print(f"  A_u = {A_u_sim:.6e}")
print(f"  A_rho = {A_rho_sim}")
print("=" * 50)
print(f"GAMER data range: a = [{a_gamer. min():.4f}, {a_gamer.max():.4f}]")
print(f"GAMER data points: {len(a_gamer)}")
print("=" * 50)