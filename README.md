# KappaLens

**Auditable atom-group analysis for lattice thermal transport**

KappaLens is a Python command-line postprocessor for comparing how selected
atom groups participate in thermal transport. It reads existing Phono3py
conductivity results, exported molecular-dynamics heat currents, and spectral
data. It does **not** run a force calculator, fit an interatomic potential, or
claim that a material has a unique “conductivity owned by each atom.”

This repository is a public **demo**. Its two toy systems, atom maps, current
traces, spectra, and optional HDF5 files are generated deterministically by
`examples/demo/generate_demo.py`. They are synthetic and have no material
identity or physical interpretation. No research structures, measurements,
trajectories, manuscripts, or server configuration are included.

## What it computes

| Command | Input | Output | Interpretation |
| --- | --- | --- | --- |
| `modes` | Phono3py `kappa-*.hdf5`, `phonon-*.hdf5`, atom-group map | Mode-by-mode weights and projected intraband conductivity | Group character of heat-carrying phonon eigenvectors |
| `gk` | Time series of total and group heat currents | All ordered group correlation integrals and block standard errors | Green–Kubo contribution under a stated energy/virial partition |
| `dsf` | Dynasor NPZ or spectrum CSV | Damped-oscillator peak position, width, fit quality | Spectral diagnostic; not a conductivity |
| `compare` | Completed summaries for multiple systems | Component tables and pairwise differences | Like-for-like comparison within one method |

The group names are user-defined: `framework`/`pendant`, `mainchain`/
`sidechain`, `linker`/`guest`, or another exhaustive partition. Each atom in a
phonon primitive cell belongs to exactly one group. A Green–Kubo current may
instead use an explicit `heat_current_groups` list without a phonon atom map.

### Scientific scope

- `modes` applies to a periodic crystal with matching Phono3py conductivity
  and eigenvector files ([1](#ref-1), [2](#ref-2)). It projects **intraband**
  `mode_kappa`; a Wigner or other interband term is outside this projection
  ([3](#ref-3), [4](#ref-4)). The program verifies that `mode_kappa`
  reconstructs the source `kappa` before reporting group results.
- `gk` can be used for ordered or disordered MD systems **if** the potential,
  per-atom energy/stress convention, heat current, trajectory, and volume have
  been validated. Its group cross terms are retained: for groups A and B,
  `AA + AB + BA + BB` reconstructs the total current correlation. Green–Kubo
  theory and atom-level correlation breakdown motivate this analysis
  ([5](#ref-5), [6](#ref-6), [7](#ref-7)); group attribution still depends on
  the chosen microscopic current convention ([8](#ref-8), [9](#ref-9),
  [10](#ref-10)).
- `dsf` fits a chosen positive, isolated spectral peak from dynamical spectra
  ([12](#ref-12)). The reported `gamma` uses the displayed frequency unit and
  is a fit parameter, not automatically a lifetime or thermal conductivity.
- A finite isolated molecule does not have the bulk conductivity tensor used
  here. A vacuum-containing 2D cell requires an explicit sheet/thickness
  normalization before comparing its reported W/(m·K) to other systems
  ([13](#ref-13)).

These references explain the underlying methods and their limitations. They
do **not** validate KappaLens against real Phono3py or MD outputs.

See the official [Phono3py HDF5 format](https://phonopy.github.io/phono3py/input-output-files.html),
[interband transport](https://phonopy.github.io/phono3py/inter-band-transport.html),
[LAMMPS heat flux](https://docs.lammps.org/compute_heat_flux.html), and
[LAMMPS units](https://docs.lammps.org/units.html) documentation for the
underlying conventions.

## Install and run the complete synthetic demo

Python 3.9 or newer is required. Work in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[demo]"
python examples/demo/generate_demo.py --require-hdf5
kappalens check --config examples/demo/project.json
kappalens modes --config examples/demo/project.json
kappalens gk --config examples/demo/project.json
kappalens dsf --config examples/demo/project.json
kappalens compare --config examples/demo/project.json --stage modes
kappalens compare --config examples/demo/project.json --stage gk
```

The generated inputs and outputs are under `examples/demo/data/` and
`examples/demo/analysis_results/`; both directories are ignored by Git. If
`h5py` is unavailable, run the generator without `--require-hdf5` and run
`gk`, `dsf`, and `compare --stage gk`. The synthetic HDF5 fixtures are only a
format demonstration, **not** genuine force-constant calculations.

Run regression tests with:

```bash
python -m unittest discover -s tests -v
```

CI runs these tests and the complete generated demo on two Python versions.

## Use your own results

Create a project JSON using `examples/demo/project.json` as a syntax reference.
Paths are resolved relative to the JSON file, so the command can be called
from any working directory. Do not copy the demo's artificial confirmation
flags or physical parameters into a research calculation without checking them.

### 1. Define atom groups for `modes`

Provide an atom map and group CSV in the **Phono3py primitive-cell atom
order**. Both files need `element` and one of `atom_index_1based` or the legacy
`poscar_index_1based` columns. The group CSV also needs `group`:

```csv
atom_index_1based,element,group
1,C,framework
2,H,pendant
```

For a new atom map, `kappalens init-groups --config project.json --model NAME`
creates an `UNASSIGNED` template without guessing chemistry or overwriting an
existing file. Fill every group manually. Set
`primitive_order_matches_atom_map: true` only after checking the actual
primitive-cell order. Set `group_star_invariant: true` only if the chosen
groups are preserved by the q-star symmetries; otherwise use a full-grid
calculation. `modes` checks mesh, q-point and branch frequencies, q weights,
temperature, atom count, imaginary modes, and reconstruction of the original
conductivity. Phono3py supports several force calculators, including VASP,
QE and LAMMPS; KappaLens reads its HDF5 outputs rather than the force
calculator's raw files.

The eigenvector weight is the sum of `|e_iα|²` over a group's components of
Phono3py's normalized dynamical-matrix eigenvector. This is mass-weighted
mode character, not the physical displacement amplitude `|e_iα|²/m_i`. The
atom-group projection is KappaLens's stated diagnostic convention; the
Phono3py papers describe the upstream phonon calculation, not an endorsement
of a unique atom-owned conductivity ([1](#ref-1), [2](#ref-2)).

### 2. Export group heat currents for `gk`

Supply a CSV with one uniform `time_ps` **or** `time_fs` column,
`Q_total_x/y/z`, and `Q_<group>_x/y/z` for every group. These are **extensive**
heat currents, before division by volume. Use `heat_current_unit`:

| Value | Source units |
| --- | --- |
| `ev_angstrom_per_ps` | eV Å/ps, e.g. LAMMPS `units metal` |
| `kcal_per_mol_angstrom_per_fs` | (kcal/mol) Å/fs, e.g. LAMMPS `units real` |

KappaLens converts the latter using `1 kcal = 4184 J` and the exact Avogadro
and elementary-charge constants. Other units, including reduced `lj` units,
must be converted before input. Set `temperature_k`, `volume_ang3`,
`directions_cart`, `gk_blocks`, `max_lag_ps`, and an inspected `plateau_ps`
window. The CSV must satisfy `Q_total = Σ Q_group` at every sample; the
integrated group terms must reproduce the direct total. The block standard
error is a diagnostic, not proof that the integral has converged. Check the
curves, trajectory length, sampling, and plateau sensitivity
([5](#ref-5), [6](#ref-6), [11](#ref-11)).

For an MD-only system, omit `atom_map` and `groups_csv` and add, for example:

```json
"heat_current_groups": ["mainchain", "sidechain"]
```

If both a group map and this list are supplied, the names must agree. A CSV
cannot prove that the simulation assigned atoms correctly; retain the MD
group-definition script and validate its heat-flux implementation. In
particular, bonded, many-body, and long-range interactions may need a
potential-specific heat-current treatment ([8](#ref-8), [9](#ref-9),
[10](#ref-10)).
For LAMMPS, inspect its current [`compute heat/flux` guidance](https://docs.lammps.org/compute_heat_flux.html)
on per-atom stress and bonded interactions before exporting group currents.

### 3. Supply spectral data for `dsf`

Use **one** of `dsf_npz` (a trusted Dynasor sample) or `dsf_csv`. The CSV
requires `q_index,q_x,q_y,q_z,omega` and one or more spectrum columns. Every
q-point must use the same ascending frequency grid. Set `dsf_omega_unit` and
one or more `dsf_fit_windows` entries with `field`, `q_index`, `omega_min`, and
`omega_max`. The fit only accepts nonnegative auto spectra; inspect whether
the chosen window really contains a single peak ([12](#ref-12)).

### 4. Inspect and compare results

Each stage writes a `summary.json` and a CSV under
`<output_dir>/<model>/<stage>/`. Summaries include resolved source paths,
input SHA-256 hashes, method name, temperature where relevant, and directions.
`compare --stage modes` and `compare --stage gk` stay separate and require
matching temperatures and direction vectors. Group labels and their meaning
must be comparable across models. Never add the Phono3py projection to the
Green–Kubo result: they are different analyses of thermal transport.

## Development and release status

This is version **0.1.0**, an alpha demo. The repository contains synthetic
tests and CI; any new reader or scientific interpretation should be validated
against real, independently checked software outputs before publication of a
material result. Contributions and issue reports should include a minimal
**synthetic or shareable** reproducer rather than unpublished research data.

Licensed under [MIT](LICENSE).

---

# 中文说明

**KappaLens：按原子组分析晶格热输运的可核查后处理程序**

KappaLens 读取已有的 Phono3py 热导结果、分组热流时间序列和振动谱数据。它
不会提交第一性原理或分子动力学任务，也不会自动生成势函数。程序中的“分组贡献”
必须结合所用方法解释，不能理解为每个原子具有唯一、独立的热导率。

本仓库是可公开的**演示版**。`examples/demo/generate_demo.py` 生成两个完全合成的
玩具体系；原子表、热流、频谱和可选的 HDF5 文件均不对应真实材料。本仓库不含
任何研究结构、实际计算数据、论文稿件或服务器配置。

## 功能与适用范围

| 命令 | 输入 | 结果与含义 |
| --- | --- | --- |
| `modes` | Phono3py 的 `kappa-*.hdf5`、`phonon-*.hdf5`、原子分组表 | 逐模态原子组权重与带内热导投影；表示导热模态的原子组特征 |
| `gk` | 总热流和各组热流的时间序列 | Green–Kubo 自关联、全部有序交叉关联、积分曲线与分块标准误 |
| `dsf` | Dynasor NPZ 或频谱 CSV | 指定峰的阻尼振子拟合参数；峰宽本身不是热导率 |
| `compare` | 多个体系已经完成的结果 | 同一方法下的分组表和体系间差值 |

原子组名称可以是 `framework`/`pendant`、`mainchain`/`sidechain` 等，不要求
某一组必须叫 `backbone`。对声子模态，每个原胞原子必须恰好归入一个组；仅做
MD 的 Green–Kubo 分析时可以直接指定 `heat_current_groups`，无需声子原子表。

`modes` 适用于具有一致的 Phono3py 热导和本征矢输出的**周期晶体**
([1](#ref-1), [2](#ref-2))，当前只投影 `mode_kappa` 对应的带内部分，
不能声称覆盖 Wigner 带间项 ([3](#ref-3), [4](#ref-4))。`gk` 可用于
有序或无序 MD 体系，但前提是势函数、逐原子能量与应力定义、热流、轨迹和体积
已得到验证 ([5](#ref-5), [6](#ref-6), [7](#ref-7), [8](#ref-8),
[9](#ref-9), [10](#ref-10))。有限孤立分子没有这里使用的体相热导张量；
含真空二维晶胞的 W/(m·K) 数值也需先明确片层厚度或面热导归一化
([13](#ref-13))。这些文献说明方法背景与限制，**不等于** KappaLens 已经通过
真实 Phono3py 或 MD 数据的验证。

## 安装并运行完整的合成示例

需要 Python 3.9 或更新版本，建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[demo]"
python examples/demo/generate_demo.py --require-hdf5
kappalens check --config examples/demo/project.json
kappalens modes --config examples/demo/project.json
kappalens gk --config examples/demo/project.json
kappalens dsf --config examples/demo/project.json
kappalens compare --config examples/demo/project.json --stage modes
kappalens compare --config examples/demo/project.json --stage gk
```

输入和结果分别生成在 `examples/demo/data/` 与
`examples/demo/analysis_results/`，这两个目录不会上传到 Git。如果尚未安装
`h5py`，可不加 `--require-hdf5` 运行生成器，先体验 `gk`、`dsf` 以及
`compare --stage gk`。演示 HDF5 只是用于验证文件接口，**不是**真正由
力常数计算得到的声子结果。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 使用自己的计算结果

以 `examples/demo/project.json` 为语法示例建立新的配置文件。相对路径以
**配置文件所在目录**为基准，与敲命令时所在目录无关。演示文件里的温度、体积、
方向和确认标志都是人工设定，不能直接套用于真实材料。

### 声子模态的原子分组

`atom_map` 与 `groups_csv` 必须采用 **Phono3py 原胞中的原子顺序**。
两者需要 `element` 以及 `atom_index_1based` 列；旧格式
`poscar_index_1based` 也可读取。分组表另需 `group` 列。可先运行：

```bash
kappalens init-groups --config project.json --model NAME
```

程序会生成标记为 `UNASSIGNED` 的模板，不会猜测化学归属，也不会覆盖已有
分组表。请逐原子填写。核对原胞顺序后才能将
`primitive_order_matches_atom_map` 设为 `true`；确认对称星操作不交换原子组
后才能将 `group_star_invariant` 设为 `true`，否则应使用完整 q 网格。
程序会检查网格、q 点、频率、权重、温度、原子数、虚频，以及投影能否还原
源文件中的带内热导。

使用的权重是组内本征矢分量 `|e_iα|²` 的和，表示质量加权坐标下的模态特征，
不同于物理位移平方 `|e_iα|²/m_i`。Phono3py 可接入多种力计算器；KappaLens
读取的是其 HDF5 结果，而不是各计算器原始输出。这个分组投影是 KappaLens
明确采用的一种诊断约定，不能把原始 Phono3py 文献理解为认可“原子独有热导率”
([1](#ref-1), [2](#ref-2))。

### 分组热流与 Green–Kubo

CSV 需要等间隔的 `time_ps` 或 `time_fs` 列、`Q_total_x/y/z`，以及各组的
`Q_<group>_x/y/z`。热流必须是**未除体积**的总热流。`heat_current_unit`
支持 `ev_angstrom_per_ps`（例如 LAMMPS `metal`）和
`kcal_per_mol_angstrom_per_fs`（例如 LAMMPS `real`）；约化 `lj` 等其他单位
需要先转换。配置中还要填写 `temperature_k`、`volume_ang3`、
`directions_cart`、`gk_blocks`、`max_lag_ps` 和检查过的 `plateau_ps` 区间。

程序要求每一帧 `Q_total = Σ Q_group`，并要求全部组间积分之和还原直接
计算的总积分。输出的分块标准误只是诊断指标，仍需检查积分平台、轨迹长度和
采样收敛 ([5](#ref-5), [6](#ref-6), [7](#ref-7), [11](#ref-11))。只做 MD 时可省去
`atom_map` 和 `groups_csv`，改用
`"heat_current_groups": ["mainchain", "sidechain"]`。CSV 无法证明 MD 引擎
给原子分组正确，因此应保存分组脚本，并验证所用势函数的热流实现；成键、多体
和长程相互作用尤其需要检查 ([8](#ref-8), [9](#ref-9), [10](#ref-10))。
使用 LAMMPS 时还应
核对其 [`compute heat/flux` 官方说明](https://docs.lammps.org/compute_heat_flux.html)
中关于逐原子应力和成键项的提示。

### 振动谱与结构间比较

`dsf` 输入选择 `dsf_npz`（可信的 Dynasor 样本）或 `dsf_csv`，不能同时
设置。CSV 需要 `q_index,q_x,q_y,q_z,omega` 及至少一列频谱。每个 q 点
应采用相同的递增频率网格；配置中需给出 `dsf_omega_unit` 和包含
`field`、`q_index`、`omega_min`、`omega_max` 的拟合窗口。先检查窗口确实
只有一个可辨识的非负自谱峰 ([12](#ref-12))。

每个阶段在 `<output_dir>/<model>/<stage>/` 输出 CSV 和 `summary.json`，
后者记录源路径、SHA-256、方法、温度及方向。`compare` 分别对 `modes` 和
`gk` 运行；两者是不同的分析方法，**不可直接相加**。跨材料比较时还须保持
组定义、温度、方向和体积约定一致。

## 开发状态

当前版本 **0.1.0** 为演示版，含合成测试和 CI。真实材料的科学结果仍需使用
独立核验过的输出进行验证。提交问题或改进建议时请使用合成或可公开的最小
示例，不要上传尚未公开的研究数据。

授权协议：[MIT](LICENSE)。

## References / 参考文献

The papers below support the method descriptions and cautions above; they are
not validation results for this demo. 使用真实结果撰写论文时，还应按实际使用的
计算软件及其版本要求引用原始方法与软件论文。

- <a id="ref-1"></a>**[1]** Togo, A., Chaput, L. & Tanaka, I. “Distributions of phonon lifetimes in Brillouin zones.” *Physical Review B* **91**, 094306 (2015). [doi:10.1103/PhysRevB.91.094306](https://doi.org/10.1103/PhysRevB.91.094306). Phono3py thermal-transport method.
- <a id="ref-2"></a>**[2]** Togo, A., Chaput, L., Tadano, T. & Tanaka, I. “Implementation strategies in phonopy and phono3py.” *Journal of Physics: Condensed Matter* **35**, 353001 (2023). [doi:10.1088/1361-648X/acd831](https://doi.org/10.1088/1361-648X/acd831). Software methods and data conventions.
- <a id="ref-3"></a>**[3]** Simoncelli, M., Marzari, N. & Mauri, F. “Unified theory of thermal transport in crystals and glasses.” *Nature Physics* **15**, 809–813 (2019). [doi:10.1038/s41567-019-0520-x](https://doi.org/10.1038/s41567-019-0520-x). Interband/coherence context.
- <a id="ref-4"></a>**[4]** Simoncelli, M., Marzari, N. & Mauri, F. “Wigner formulation of thermal transport in solids.” *Physical Review X* **12**, 041011 (2022). [doi:10.1103/PhysRevX.12.041011](https://doi.org/10.1103/PhysRevX.12.041011). Limits of an intraband-only interpretation.
- <a id="ref-5"></a>**[5]** Green, M. S. “Markoff random processes and the statistical mechanics of time-dependent phenomena. II. Irreversible processes in fluids.” *Journal of Chemical Physics* **22**, 398–413 (1954). [doi:10.1063/1.1740082](https://doi.org/10.1063/1.1740082). Correlation-function foundation.
- <a id="ref-6"></a>**[6]** Kubo, R. “Statistical-mechanical theory of irreversible processes. I.” *Journal of the Physical Society of Japan* **12**, 570–586 (1957). [doi:10.1143/JPSJ.12.570](https://doi.org/10.1143/JPSJ.12.570). Linear response and Green–Kubo foundation.
- <a id="ref-7"></a>**[7]** Manjunatha, L., Takamatsu, H. & Cannon, J. J. “Atomic-level breakdown of Green–Kubo relations provides new insight into the mechanisms of thermal conduction.” *Scientific Reports* **11**, 5597 (2021). [doi:10.1038/s41598-021-84446-9](https://doi.org/10.1038/s41598-021-84446-9). Example of current cross-correlation breakdown in molecular systems; not a validation of KappaLens.
- <a id="ref-8"></a>**[8]** Ercole, L., Marcolongo, A., Umari, P. & Baroni, S. “Gauge invariance of thermal transport coefficients.” *Journal of Low Temperature Physics* **185**, 79–86 (2016). [doi:10.1007/s10909-016-1617-6](https://doi.org/10.1007/s10909-016-1617-6). Microscopic-current convention and interpretation.
- <a id="ref-9"></a>**[9]** Fan, Z. *et al.* “Force and heat current formulas for many-body potentials in molecular dynamics simulations with applications to thermal conductivity calculations.” *Physical Review B* **92**, 094301 (2015). [doi:10.1103/PhysRevB.92.094301](https://doi.org/10.1103/PhysRevB.92.094301). Many-body heat-current formulas.
- <a id="ref-10"></a>**[10]** Surblys, D., Matsubara, H., Kikugawa, G. & Ohara, T. “Application of atomic stress to compute heat flux via molecular dynamics for systems with many-body interactions.” *Physical Review E* **99**, 051301(R) (2019). [doi:10.1103/PhysRevE.99.051301](https://doi.org/10.1103/PhysRevE.99.051301). Bonded/many-body stress caveat.
- <a id="ref-11"></a>**[11]** Ercole, L., Marcolongo, A. & Baroni, S. “Accurate thermal conductivities from optimally short molecular dynamics simulations.” *Scientific Reports* **7**, 15835 (2017). [doi:10.1038/s41598-017-15843-2](https://doi.org/10.1038/s41598-017-15843-2). Green–Kubo estimation and uncertainty context.
- <a id="ref-12"></a>**[12]** Fransson, E., Slabanja, M., Erhart, P. & Wahnström, G. “dynasor—A tool for extracting dynamical structure factors and current correlation functions from molecular dynamics simulations.” *Advanced Theory and Simulations* **4**, 2000240 (2021). [doi:10.1002/adts.202000240](https://doi.org/10.1002/adts.202000240). Spectral analysis context.
- <a id="ref-13"></a>**[13]** Wu, X. *et al.* “How to characterize thermal transport capability of 2D materials fairly? Sheet thermal conductance and the choice of thickness.” *Chemical Physics Letters* **669**, 233–237 (2017). [doi:10.1016/j.cplett.2016.12.054](https://doi.org/10.1016/j.cplett.2016.12.054). Thickness convention for 2D comparisons.

Software format details should be checked against the current [Phono3py HDF5 documentation](https://phonopy.github.io/phono3py/input-output-files.html), [Phono3py citation guidance](https://phonopy.github.io/phono3py/citation.html), and [LAMMPS heat-flux documentation](https://docs.lammps.org/compute_heat_flux.html).
