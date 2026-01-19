"""
代码名：convergence_compare_at_a.py

使用方法：
1) 在本文件开头的“可调参数区”里：
   - 设置 A_TARGETS（指定一个或多个要比较的尺度因子 a）
   - 在 DATA_FILES 里添加/删除要对比的 txt 文件路径（可无限添加）
   - 如有需要，调整解析解参数（k/H0/gamma/ai/初始扰动幅度等）
2) 直接运行：
   python convergence_compare_at_a.py

代码功能：
- 调用本仓库的解析/数值求解器 ScipyComovingMagnetosonicWave 计算理论值 δρc/ρc(a)
- 从一个或多个 GAMER 导出的 txt 文件中提取数组 a_values 与 delta_rho_norm_lefthalf
- 在指定的 a（或多个 a）处，把“数据值”与“理论值”相除得到 ratio，并计算相对误差：ratio - 1
- 将每个输入文件在目标 a 处的误差在同一幅图中画出，并在命令行输出每个文件的误差数值

实现说明（要点）：
- 不使用 exec：用正则提取 `xxx = np.array([ ... ])` 的括号内容，再用 np.fromstring 安全解析。
- 支持两种取值方式：
  - LOG_INTERP：在 log(a) 空间对 y(a) 做线性插值（更适合跨数量级的 a）
  - NEAREST：取最近的离散采样点
- 当理论值非常接近 0 时，相对误差会不稳定：此时会输出 absolute_error，并将相对误差记为 NaN。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from comoving_mhd_waves import ScipyComovingMagnetosonicWave


# ============================================================
# 可调参数区（按需改这里就行）
# ============================================================

# 1) 你要比较的目标尺度因子 a（可写一个，也可写多个）
A_TARGETS = np.array([
    1.0,
    # 0.20,
    # 0.30,
])

# 2) 要读取并比较的 txt 文件（可无限添加）
#    - 建议写相对路径：以“本脚本所在目录”为基准
DATA_FILES: list[tuple[str, str]] = [
    ("deltarho_norm_data_lefthalf-MHM_RP-CFL0.1.txt", "MHM_RP CFL0.1"),
    ("deltarho_norm_data_lefthalf-MHM_RP-CFL0.3.txt", "MHM_RP CFL0.3"),
    # ("deltarho_norm_data_lefthalf-MHM-CFL0.1.txt", "MHM CFL0.1"),
]

# 3) 选择对比哪一部分（你的数据是实数，一般对应理论的 real 部分）
THEORY_COMPONENT: Literal["real", "imag", "abs"] = "real"

# 4) 从数据数组上取 y(a_target) 的方式
SAMPLE_METHOD: Literal["log_interp", "nearest"] = "log_interp"

# 5) 解析解参数（沿用你现有脚本的默认设置，可自行修改）
K = 2.0 * np.pi
H0 = 1.0
GAMMA = 5.0 / 3.0
AI = 1.0 / 128.0

RHO_SIM = 1.0
P_SIM = 1.0
BZ_SIM = 1.0

# 初始扰动幅度（和你现有对比脚本一致：给一个很小的速度扰动）
# 注意：ScipyComovingMagnetosonicWave 内部把 y[1] 设为 ai*A_u*Vs，且 RHS 用的是 complex。
A_U_SIM = 1j * (1.0e-5 / RHO_SIM) / np.sqrt(GAMMA * P_SIM / RHO_SIM) / AI
A_RHO_SIM = 0.0

# 理论值过小判定阈值（避免 ratio 爆炸）
THEORY_EPS = 1e-30

# 输出/作图控制
SAVE_FIG = True
FIG_NAME = "convergence_compare_at_a.png"


# ============================================================
# 工具函数
# ============================================================


def _as_float_array_from_np_array_text(content: str, varname: str) -> np.ndarray:
    """从类似 `var = np.array([ ... ])` 的文本中解析出 float 数组。"""
    # 兼容空格/换行
    pattern = rf"{re.escape(varname)}\s*=\s*np\.array\(\s*\[(.*?)\]\s*\)"
    match = re.search(pattern, content, flags=re.S)
    if match is None:
        raise ValueError(f"在文件内容中找不到变量 {varname} 的 np.array([...]) 定义")

    body = match.group(1)
    # 说明：之前用 np.fromstring(sep=",") 在个别环境下会把科学计数法的指数部分误当成独立数字
    # （例如把 e-01 的 -01 解析成额外的 -1），导致 a_values 出现非物理的负值。
    # 这里改用正则显式提取浮点数，确保稳定。
    float_pattern = re.compile(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    )
    tokens = float_pattern.findall(body)
    arr = np.array([float(t) for t in tokens], dtype=float)
    if arr.size == 0:
        raise ValueError(f"变量 {varname} 解析结果为空数组，可能是格式不匹配")
    return arr


@dataclass(frozen=True)
class GamerSeries:
    a: np.ndarray
    y: np.ndarray


def load_gamer_txt(path: Path) -> GamerSeries:
    """读取 GAMER 导出的 txt（内含 a_values / delta_rho_norm_lefthalf 的 np.array 定义）。"""
    content = path.read_text(encoding="utf-8")
    a_values = _as_float_array_from_np_array_text(content, "a_values")
    y_values = _as_float_array_from_np_array_text(content, "delta_rho_norm_lefthalf")

    if a_values.shape != y_values.shape:
        raise ValueError(
            f"a_values 与 delta_rho_norm_lefthalf 长度不一致：{a_values.size} vs {y_values.size}"
        )

    return GamerSeries(a=a_values, y=y_values)


def _sample_y_at_targets(
    series: GamerSeries,
    a_targets: np.ndarray,
    method: Literal["log_interp", "nearest"],
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (a_used, y_used)。

    - log_interp：在 log(a) 上插值，a_used = a_targets
    - nearest：取最近点，a_used 为数据中最近点的 a
    """
    a = np.asarray(series.a, dtype=float)
    y = np.asarray(series.y, dtype=float)

    order = np.argsort(a)
    a_sorted = a[order]
    y_sorted = y[order]

    if method == "nearest":
        a_used = np.empty_like(a_targets, dtype=float)
        y_used = np.empty_like(a_targets, dtype=float)
        for i, at in enumerate(a_targets):
            idx = int(np.argmin(np.abs(a_sorted - at)))
            a_used[i] = a_sorted[idx]
            y_used[i] = y_sorted[idx]
        return a_used, y_used

    if method == "log_interp":
        if np.any(a_sorted <= 0):
            raise ValueError("log_interp 需要 a_values 全部为正")

        x = np.log(a_sorted)
        xt = np.log(a_targets)
        # np.interp 要求 x 单调递增，已 sort
        y_used = np.interp(xt, x, y_sorted)
        return np.array(a_targets, dtype=float), y_used

    raise ValueError(f"未知 method={method}")


def _select_component(z: np.ndarray, component: Literal["real", "imag", "abs"]) -> np.ndarray:
    if component == "real":
        return np.real(z)
    if component == "imag":
        return np.imag(z)
    if component == "abs":
        return np.abs(z)
    raise ValueError(f"未知 component={component}")


# ============================================================
# 主流程
# ============================================================


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    # 构建理论解
    vs_sim = np.sqrt(GAMMA * P_SIM / RHO_SIM)
    va_sim = BZ_SIM / np.sqrt(RHO_SIM)
    vg_sim = np.sqrt(1.5) * H0 / K

    sol = ScipyComovingMagnetosonicWave(
        K,
        H0,
        vs_sim,
        va_sim,
        vg_sim,
        GAMMA,
        AI,
        A_U_SIM,
        A_RHO_SIM,
    )

    a_targets = np.asarray(A_TARGETS, dtype=float)
    if a_targets.ndim != 1 or a_targets.size < 1:
        raise ValueError("A_TARGETS 必须是一维且至少包含一个元素")

    # 理论值（按指定 component）
    theory_complex = sol.delta_rhoc_over_rhoc(a_targets)
    theory = _select_component(theory_complex, THEORY_COMPONENT).astype(float)

    print("=" * 72)
    print("Convergence / Error check at specified a")
    print(f"  component      = {THEORY_COMPONENT}")
    print(f"  sample_method  = {SAMPLE_METHOD}")
    print(f"  a_targets      = {a_targets}")
    print("  parameters:")
    print(f"    gamma={GAMMA}, ai={AI}, k={K}, H0={H0}")
    print(f"    Vs={vs_sim}, Va={va_sim}, Vg={vg_sim}")
    print("=" * 72)

    results_per_file: dict[str, dict[str, np.ndarray]] = {}

    for rel_path, label in DATA_FILES:
        path = (script_dir / rel_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"找不到数据文件：{rel_path}（解析到：{path}）")

        series = load_gamer_txt(path)
        a_used, data_y = _sample_y_at_targets(series, a_targets, SAMPLE_METHOD)

        # ratio = data / theory
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = data_y / theory

        # 相对误差：data/theory - 1。理论过小则置 NaN，并提供绝对误差。
        abs_error = data_y - theory
        rel_error = np.full_like(ratio, np.nan, dtype=float)
        mask = np.abs(theory) > THEORY_EPS
        rel_error[mask] = ratio[mask] - 1.0

        results_per_file[label] = {
            "a_used": a_used,
            "data": data_y,
            "theory": theory,
            "ratio": ratio,
            "rel_error": rel_error,
            "abs_error": abs_error,
        }

        # 命令行输出
        print(f"[{label}]  file={path.name}")
        for i, at in enumerate(a_targets):
            au = a_used[i]
            dy = data_y[i]
            th = theory[i]
            ra = ratio[i]
            re = rel_error[i]
            ae = abs_error[i]
            if np.isfinite(re):
                print(
                    f"  a_target={at:.6g}  a_used={au:.6g}  data={dy:+.6e}  theory={th:+.6e}  "
                    f"ratio=data/theory={ra:+.6e}  rel_err=ratio-1={re:+.6e}"
                )
            else:
                print(
                    f"  a_target={at:.6g}  a_used={au:.6g}  data={dy:+.6e}  theory={th:+.6e}  "
                    f"theory≈0 => abs_err=data-theory={ae:+.6e} (rel_err=NaN)"
                )
        print("-" * 72)

    # 作图：
    # - 若只给一个 a：画每个文件的 |rel_error| 柱状图
    # - 若多个 a：画每个文件随 a 的 |rel_error(a)| 曲线
    fig = plt.figure(figsize=(10, 6))

    if a_targets.size == 1:
        labels = list(results_per_file.keys())
        yvals = []
        for label in labels:
            re = results_per_file[label]["rel_error"][0]
            if np.isfinite(re):
                yvals.append(abs(re))
            else:
                yvals.append(np.nan)

        x = np.arange(len(labels), dtype=float)
        plt.bar(x, yvals)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel(r"$|\mathrm{rel\_err}| = |\mathrm{data/theory}-1|$")
        plt.title(f"Relative error at a={a_targets[0]:.6g} ({THEORY_COMPONENT})")
        plt.grid(True, axis="y", alpha=0.3)

    else:
        for label, r in results_per_file.items():
            re = r["rel_error"]
            plt.semilogx(a_targets, np.abs(re), marker="o", linewidth=1.8, label=label)

        plt.xlabel("Scale factor a")
        plt.ylabel(r"$|\mathrm{rel\_err}(a)| = |\mathrm{data/theory}-1|$")
        plt.title(f"Relative error vs a ({THEORY_COMPONENT}, {SAMPLE_METHOD})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(loc="best")

    plt.tight_layout()

    if SAVE_FIG:
        out = (script_dir / FIG_NAME).resolve()
        fig.savefig(out, dpi=160)
        print(f"Figure saved: {out}")

    plt.show()


if __name__ == "__main__":
    main()
