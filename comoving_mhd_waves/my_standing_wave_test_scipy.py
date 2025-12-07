import numpy as np
import matplotlib.pyplot as plt
from comoving_mhd_waves import ScipyComovingMagnetosonicWave

# 你的参数
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

A_u_sim = (1.0e-5 / rho_sim) / Vs_sim
A_rho_sim = 0.0

# 使用 Scipy 数值求解器（更稳定）
sol = ScipyComovingMagnetosonicWave(k, H0, Vs_sim, Va_sim, Vg_sim, gamma, ai, A_u_sim, A_rho_sim)

# 画图
a = np.logspace(np.log10(ai), 0, 300)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), sharex=True)

axes[0, 0].semilogx(a, sol.delta_rhoc_over_rhoc(a). real)
axes[0, 0]. set_title(r'$\mathrm{Re}(\delta \rho_c/\rho_c)$')

axes[0, 1].semilogx(a, sol.delta_rhoc_over_rhoc(a).imag)
axes[0, 1].set_title(r'$\mathrm{Im}(\delta \rho_c/\rho_c)$')

axes[1, 0].semilogx(a, sol.delta_u_over_vs(a).real)
axes[1, 0].set_title(r'$\mathrm{Re}(\delta u/V_s)$')
axes[1, 0].set_xlabel('Scale factor a')

axes[1, 1].semilogx(a, sol. delta_u_over_vs(a). imag)
axes[1, 1].set_title(r'$\mathrm{Im}(\delta u/V_s)$')
axes[1, 1].set_xlabel('Scale factor a')

plt.tight_layout()
plt.show()