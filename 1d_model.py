import math
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
#                  Tug-of-War Rope Model
#  - Ground: traction-limited drive + rolling resistance
#  - Gait: half-wave sine activation a(phi)=max(0, sin(phi))
#  - Rope: tension-only springs + Kelvin–Voigt damping per segment
#  - Agents: Kuramoto phase coupling within each team
# ============================================================

# -------------------- DISCRETIZATION & PLACEMENT --------------------
N = 100
Nagents = 2            # robots per team
FIRST_OFFSET_CM = 30.0      # innermost robot distance from center (per side)

# -------------------- PHYSICAL PARAMETERS (for ND scaling) --------------------
M_robot = 2.9        # kg
mu_s    = 0.4
g       = 9.81       # m/s^2

# Rope link construction
k_spring = 396.0     # N/m per spring
L_spring = 0.10      # m
m_spring = 0.06      # kg
n_series   = 3
n_parallel = 2

k_series = k_spring / n_series
k_link   = n_parallel * k_series
L_link   = n_series * L_spring
m_link   = n_parallel * n_series * m_spring

EA     = k_link * L_link          # force scale (N)

# Rope geometry
L_rope_phys = 1.5                 # m
l0_phys     = L_rope_phys / (N - 1)  # rest length per segment (m)
refLen      = l0_phys
L_rope_nd   = L_rope_phys / refLen
dx_nd       = 1

rho_rope= m_link / L_link        # mass per length (kg/m)
m0      = rho_rope * refLen                   # mass scale (kg)

# ND scales
T_star  = math.sqrt(m0 * refLen / EA)
lam     = M_robot / m0
sigma_s = mu_s * M_robot * g / EA          # ND traction limit

# -------------------- OSCILLATORS (Kuramoto) --------------------
rpm_motor  = 9.5
omega_phys = 2.0 * math.pi * rpm_motor / 60.0
W0         = omega_phys * T_star

omega_std   = 0.005
sigma_phase = 0.005

K0      = 0.5
K_left  = K0
K_right = K0

# -------------------- ACTUATION ENVELOPE --------------------
f0_factor     = 0.5
f0            = f0_factor * sigma_s
f0_noise_rel  = 0.005
f0_noise_std  = f0_noise_rel * f0

f_hold_rel = 0.30
f_hold     = f_hold_rel * sigma_s

tau_motor_phys = 0.6
tau_motor      = tau_motor_phys / T_star

tau_amp_phys = 1.00
tau_amp      = tau_amp_phys / T_star

f_cap = 1e18

# Persistent mismatch
strength_std  = 0.0
team_bias_rel = -0.02   # e.g. +0.02 makes RIGHT team stronger

# -------------------- DAMPING / RESISTANCE --------------------
beta_rope  = 0.2
gamma_phys = 0.4
zeta       = gamma_phys * T_star / m0

sigma_roll_rel = 0.05
sigma_roll     = sigma_roll_rel * sigma_s

v0_roll_phys = 0.01
v0_roll      = v0_roll_phys * T_star / refLen

# -------------------- TIME --------------------
dtau_init = 1e-2
dtau_min  = 1e-4
dtau_max  = 5e-2

tau_end_seconds      = 150.0
tau_end              = tau_end_seconds / T_star

sample_every_seconds = 0.05
sample_every         = sample_every_seconds / T_star


# ============================================================
#                       NUMBA HELPERS
# ============================================================

@njit
def init_x_linear_centered(Nloc):
    idx = np.arange(Nloc, dtype=np.float64)
    return (idx - 0.5 * (Nloc - 1)) * dx_nd

@njit
def spring_forces_tension_only_kv(x, v, beta):
    Nloc = x.shape[0]
    F = np.zeros_like(x)
    for i in range(Nloc - 1):
        e = x[i+1] - x[i] - dx_nd
        if e > 0.0:
            t = (e / dx_nd) + beta * (v[i+1] - v[i])
            if t < 0.0:
                t = 0.0
            F[i]   += t
            F[i+1] -= t
    return F

@njit
def activation_halfwave_sine(phi):
    s = math.sin(phi)
    return s if s > 0.0 else 0.0

@njit
def agent_interaction_nd(omega_tilde, phi, dtau,
                         K_left_tilde, K_right_tilde,
                         team_labels, sigma_tilde):
    n = phi.shape[0]
    phi_interaction = np.zeros(n)

    for i in range(n):
        s_i = team_labels[i]
        K = K_left_tilde if s_i == -1 else K_right_tilde
        acc = 0.0
        cnt = 0
        for j in range(n):
            if i == j:
                continue
            if team_labels[j] == s_i:
                acc += K * math.sin(phi[j] - phi[i])
                cnt += 1
        if cnt > 0:
            phi_interaction[i] = acc / cnt

    noise = sigma_tilde * np.random.normal(0.0, 1.0, n) * math.sqrt(dtau)
    return (phi + (omega_tilde + phi_interaction) * dtau + noise) % (2.0 * math.pi)

@njit
def actuation_step_nd(phi, agent_indices, team_signs,
                     amp_state, motor_state,
                     strength, team_labels,
                     team_bias_rel_local,
                     f_hold_local, f_cap_local,
                     f0_mean, f0_std,
                     tau_amp_local, tau_motor_local,
                     dtau, Nloc):
    f = np.zeros(Nloc)
    n_agents = len(agent_indices)

    if tau_amp_local > 0.0 and dtau > 0.0:
        a = dtau / tau_amp_local
        noise_scale = f0_std * math.sqrt(2.0 * dtau / tau_amp_local)
        for k in range(n_agents):
            amp_state[k] += (f0_mean - amp_state[k]) * a + noise_scale * np.random.normal(0.0, 1.0)
            if amp_state[k] < 0.0:
                amp_state[k] = 0.0

    if tau_motor_local <= 0.0:
        tau_motor_local = 1e-12
    alpha = dtau / tau_motor_local if dtau > 0.0 else 1.0

    for k in range(n_agents):
        act = activation_halfwave_sine(phi[k])
        bias = 1.0 + (team_bias_rel_local if team_labels[k] == 1 else 0.0)
        cmd = (f_hold_local + amp_state[k] * act) * strength[k] * bias

        if cmd > f_cap_local:
            cmd = f_cap_local
        if cmd < 0.0:
            cmd = 0.0

        motor_state[k] += (cmd - motor_state[k]) * alpha

        i = agent_indices[k]
        s = team_signs[k]
        f[i] = s * motor_state[k]

    return f

@njit
def apply_traction_limit_inplace(f, agent_indices, sigma_max):
    for k in range(len(agent_indices)):
        i = agent_indices[k]
        Fi = f[i]
        if Fi > sigma_max:
            f[i] = sigma_max
        elif Fi < -sigma_max:
            f[i] = -sigma_max
    return f

@njit
def rolling_resistance_nd(v, occupancy, sigma_roll_local, v0_local):
    F = np.zeros_like(v)
    for i in range(v.shape[0]):
        if occupancy[i] == 0:
            continue
        vi = v[i]
        F[i] = -sigma_roll_local * (vi / (abs(vi) + v0_local))
    return F


# ============================================================
#                      PLACEMENT
# ============================================================

def place_agents_by_index(
    Nloc,
    Nagents,
    first_offset_cm=50.0,
    spacing_cm=None,
    fill_to_ends=True,
    round_mode="nearest",
):
    if Nagents < 1:
        raise ValueError("Nagents must be >= 1.")

    half_len_m = 0.5 * L_rope_nd * refLen
    mid = 0.5 * (Nloc - 1)

    first_offset_m = first_offset_cm / 100.0

    if spacing_cm is None and fill_to_ends and Nagents > 1:
        spacing_m = (half_len_m - first_offset_m) / float(Nagents - 1)
    elif spacing_cm is None:
        spacing_m = 0.30
    else:
        spacing_m = spacing_cm / 100.0

    max_needed = first_offset_m + (Nagents - 1) * spacing_m
    if max_needed > half_len_m + 1e-12:
        raise ValueError(
            f"Requested layout exceeds half rope length: farthest={max_needed:.3f} m, half_len={half_len_m:.3f} m."
        )

    distances_m = first_offset_m + spacing_m * np.arange(Nagents, dtype=np.float64)
    offsets_idx = (distances_m / refLen) / dx_nd

    base_left   = mid - offsets_idx
    base_right  = mid + offsets_idx

    if round_mode == "nearest":
        left_init   = np.rint(base_left)
        right_init  = np.rint(base_right)
    elif round_mode == "floor":
        left_init   = np.floor(base_left)
        right_init  = np.ceil(base_right)
    elif round_mode == "ceil":
        left_init   = np.ceil(base_left)
        right_init  = np.floor(base_right)
    else:
        raise ValueError("round_mode must be 'nearest', 'floor', or 'ceil'.")

    left_init   = left_init.astype(np.int64)
    right_init  = right_init.astype(np.int64)

    def assign_with_push(base_idxs, side):
        used = set()
        out  = np.empty_like(base_idxs, dtype=np.int64)
        step = -1 if side == "left" else +1
        for k, idx in enumerate(base_idxs):
            j = int(idx)
            while (j in used) or (j < 0) or (j >= Nloc):
                j += step
                if j < 0 or j >= Nloc:
                    raise RuntimeError(f"Cannot place {side} robot {k} (out of nodes).")
            used.add(j)
            out[k] = j
        return np.sort(out).astype(np.int32)

    left_team_indices  = assign_with_push(left_init,  "left")
    right_team_indices = assign_with_push(right_init, "right")
    return left_team_indices, right_team_indices


# ============================================================
#                      SIMULATION
# ============================================================

def run_simulation_nd(seed=42, save_path="tug_of_war_nd_single_run_clean.npz", verbose=False):
    if seed is not None:
        np.random.seed(seed)

    x = init_x_linear_centered(N)
    v = np.zeros_like(x)

    L_idx, R_idx = place_agents_by_index(N, Nagents, first_offset_cm=FIRST_OFFSET_CM)
    agent_indices = np.concatenate([L_idx, R_idx]).astype(np.int32)

    team_labels = np.concatenate([
        -np.ones(len(L_idx), dtype=np.int8),
        +np.ones(len(R_idx), dtype=np.int8)
    ])
    team_signs = team_labels.copy()

    xL_front_prev = np.max(x[L_idx])
    xR_front_prev = np.min(x[R_idx])

    omega_left_agents  = np.random.normal(W0, omega_std, len(L_idx))
    omega_right_agents = np.random.normal(W0, omega_std, len(R_idx))
    omega_agents = np.concatenate([omega_left_agents, omega_right_agents])
    phi = np.random.uniform(0.0, 2.0 * math.pi, len(omega_agents))

    lam_array = np.ones(N, dtype=np.float64)
    lam_array[agent_indices] = lam
    occupancy = np.zeros(N, dtype=np.int8)
    occupancy[agent_indices] = 1

    strength = 1.0 + strength_std * np.random.normal(0.0, 1.0, len(agent_indices))
    strength = np.maximum(0.2, strength).astype(np.float64)

    amp_state   = f0 + f0_noise_std * np.random.normal(0.0, 1.0, len(agent_indices))
    amp_state   = np.maximum(0.0, amp_state).astype(np.float64)
    motor_state = np.full(len(agent_indices), f_hold, dtype=np.float64)

    tau  = 0.0
    dtau = dtau_init

    next_sample = 0.0
    times_tau = []
    x_history = []
    agent_forces = []
    agent_phases = []
    agent_velocities = []
    rope_mid = []
    rope_mid_vel = []

    if N % 2 == 0:
        mid_left  = N // 2 - 1
        mid_right = mid_left + 1
        def mid_val(arr):
            return 0.5 * (arr[mid_left] + arr[mid_right])
    else:
        mid_center = N // 2
        def mid_val(arr):
            return arr[mid_center]

    def log_state(f_applied):
        times_tau.append(tau)
        x_history.append(x.copy())
        agent_forces.append(f_applied[agent_indices].copy())
        agent_phases.append(phi.copy())
        agent_velocities.append(v[agent_indices].copy())
        rope_mid.append(mid_val(x))
        rope_mid_vel.append(mid_val(v))

    Fs = spring_forces_tension_only_kv(x, v, beta_rope)
    f_cmd = actuation_step_nd(
        phi, agent_indices, team_signs,
        amp_state, motor_state,
        strength, team_labels,
        team_bias_rel,
        f_hold, f_cap,
        f0, f0_noise_std,
        tau_amp, tau_motor,
        dtau_init, N
    )
    f = apply_traction_limit_inplace(f_cmd, agent_indices, sigma_s)
    F_roll = rolling_resistance_nd(v, occupancy, sigma_roll, v0_roll)

    Ftot = Fs - zeta * v + f + F_roll
    a = Ftot / lam_array

    log_state(f)
    next_sample += sample_every

    winner = 0

    while tau < tau_end:
        phi = agent_interaction_nd(omega_agents, phi, dtau, K_left, K_right, team_labels, sigma_phase)

        v_half = v + 0.5 * a * dtau
        x_new  = x + v_half * dtau

        Fs_new = spring_forces_tension_only_kv(x_new, v_half, beta_rope)

        f_cmd_new = actuation_step_nd(
            phi, agent_indices, team_signs,
            amp_state, motor_state,
            strength, team_labels,
            team_bias_rel,
            f_hold, f_cap,
            f0, f0_noise_std,
            tau_amp, tau_motor,
            dtau, N
        )
        f_new = apply_traction_limit_inplace(f_cmd_new, agent_indices, sigma_s)
        F_roll_new = rolling_resistance_nd(v_half, occupancy, sigma_roll, v0_roll)

        Ftot_new = Fs_new - zeta * v_half + f_new + F_roll_new
        a_new = Ftot_new / lam_array
        v_new = v_half + 0.5 * a_new * dtau

        xL_front_now = np.max(x_new[L_idx])
        xR_front_now = np.min(x_new[R_idx])
        cross_left  = (xL_front_prev <= 0.0) and (xL_front_now >= 0.0)
        cross_right = (xR_front_prev >= 0.0) and (xR_front_now <= 0.0)

        if cross_left or cross_right:
            x, v, a = x_new, v_new, a_new
            tau += dtau
            winner = +1 if cross_left else -1
            log_state(f_new)
            if verbose:
                side = "Right" if winner == 1 else "Left"
                print(f"{side} team wins at t = {tau*T_star:.3f} s")
            break

        x, v, a = x_new, v_new, a_new
        tau += dtau
        xL_front_prev = xL_front_now
        xR_front_prev = xR_front_now

        amax = np.max(np.abs(a_new))
        if amax > 1e4:
            dtau = max(dtau / 1.2, dtau_min)
        elif amax < 1e2:
            dtau = min(dtau * 1.2, dtau_max)

        if tau >= next_sample:
            log_state(f_new)
            next_sample += sample_every

    times_tau_arr = np.array(times_tau, dtype=np.float64)
    times_s = times_tau_arr * T_star

    out = {
        "times_tau": times_tau_arr,
        "times_s": times_s,
        "x_history": np.array(x_history),
        "agent_indices": agent_indices,
        "team_labels": team_labels,
        "agent_forces": np.array(agent_forces),
        "agent_phases": np.array(agent_phases),
        "agent_velocities": np.array(agent_velocities),
        "rope_mid": np.array(rope_mid),
        "rope_mid_vel": np.array(rope_mid_vel),
        "EA": EA, "m0": m0, "refLen": refLen, "T_star": T_star,
        "L_rope_phys": L_rope_phys, "L_rope_nd": L_rope_nd, "dx_nd": dx_nd,
        "lambda": lam, "sigma_s": sigma_s,
        "zeta": zeta, "beta_rope": beta_rope,
        "f0": f0, "f0_noise_std": f0_noise_std, "f_hold": f_hold,
        "tau_amp": tau_amp, "tau_motor": tau_motor,
        "rpm_motor": rpm_motor,
        "K_left": K_left, "K_right": K_right,
        "sigma_phase": sigma_phase,
        "Nagents": Nagents, "FIRST_OFFSET_CM": FIRST_OFFSET_CM,
        "strength_std": strength_std, "team_bias_rel": team_bias_rel,
        "winner": winner,
    }

    if save_path:
        np.savez(save_path, **out)
    return out


# ============================================================
#                      ANIMATION GENERATOR
# ============================================================

def animate_simulation_from_dict(sim_data, output_gif="tug_of_war.gif", skip=5, fps=10):
    plt.style.use("default")

    q_history = -1.0 * (sim_data["x_history"] * sim_data["refLen"])[::skip]
    agent_phases = sim_data["agent_phases"][::skip]
    agent_indices = sim_data["agent_indices"]
    team_labels = sim_data["team_labels"]

    left_mask = team_labels == -1
    right_mask = team_labels == 1
    left_agent_indices = agent_indices[left_mask]
    right_agent_indices = agent_indices[right_mask]

    Nsteps, N_nodes = q_history.shape

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="white")
    ax.set_facecolor("white")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    x_min = np.min(q_history) - 0.2
    x_max = np.max(q_history) + 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, 0.2)

    rope_line, = ax.plot([], [], "-", linewidth=2, color="black", zorder=10)
    ax.vlines(0, -0.1, 0.1, linewidth=2, color="gray", linestyle="--", zorder=5)

    team_left_scatter = ax.scatter(
        [], [], s=200, facecolors="white", edgecolors="tab:blue", linewidths=1.5, zorder=15
    )
    team_right_scatter = ax.scatter(
        [], [], s=200, facecolors="white", edgecolors="tab:red", linewidths=1.5, zorder=15
    )
    center_scatter = ax.scatter([], [], s=60, color="black", zorder=20)

    def update(timeStep):
        q = q_history[timeStep, :]
        phi = agent_phases[timeStep, :]

        phi_mod = phi % (2 * np.pi)
        pulling = np.maximum(0.0, np.sin(phi_mod)) > 0.0

        left_facecolors = np.where(pulling[left_mask], "tab:blue", "white")
        right_facecolors = np.where(pulling[right_mask], "tab:red", "white")

        pos_left = q[left_agent_indices]
        pos_right = q[right_agent_indices]

        mid_idx = N_nodes // 2
        center_pos = 0.5 * (q[mid_idx - 1] + q[mid_idx]) if N_nodes % 2 == 0 else q[mid_idx]

        center_scatter.set_offsets(np.c_[center_pos, 0])
        team_left_scatter.set_offsets(np.c_[pos_left, np.zeros_like(pos_left)])
        team_right_scatter.set_offsets(np.c_[pos_right, np.zeros_like(pos_right)])

        team_left_scatter.set_facecolors(left_facecolors)
        team_right_scatter.set_facecolors(right_facecolors)
        rope_line.set_data(q, np.zeros_like(q))

        return rope_line, team_left_scatter, team_right_scatter, center_scatter

    anim = FuncAnimation(fig, update, frames=Nsteps, interval=1000/fps, blit=False)
    plt.tight_layout()
    writer = PillowWriter(fps=fps)
    anim.save(output_gif, writer=writer)
    plt.close()
    print(f"GIF animation successfully saved to {output_gif}")


# ---------------------- EXECUTION ----------------------
if __name__ == "__main__":
    result = run_simulation_nd(seed=42, save_path="tug_of_war_nd_single_run_clean1.npz", verbose=True)
    print("winner:", result["winner"])

    # Generate GIF animation directly from sim output
    animate_simulation_from_dict(result, output_gif="tug_of_war.gif", skip=5, fps=10)