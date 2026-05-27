import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from dataclasses import dataclass

g0 = 9.80665
Re = 6371e3
mu = 3.986004418e14
R_air = 287.05


def atmosphere_isa(h):
    h = max(h, 0.0)

    if h <= 11000:
        T = 288.15 - 0.0065*h
        p = 101325 * (T/288.15)**(g0/(-0.0065*R_air))
    elif h <= 20000:
        T = 216.65
        p11 = 22632.06
        p = p11 * np.exp(-g0*(h-11000)/(R_air*T))
    else:
        T = 216.65
        p20 = 5474.889
        p = p20 * np.exp(-g0*(h-20000)/(R_air*T))

    rho = p/(R_air*T)
    return T, p, rho


def gravity(h):
    h = max(h, 0.0)
    return mu/(Re + h)**2


@dataclass
class Rocket:
    m0: float
    mf: float
    diameter: float
    Cd: float | None = None
    Cl: float | None = None

    @property
    def area_ref(self):
        return 0.25*np.pi*self.diameter**2


def load_thrust_curve_excel(filename, time_col="t", thrust_col="T"):
    import os

    file_path = os.path.join("ALL_BEM_2024_2025", "empuje", filename)

    data = pd.read_csv(file_path, sep=",", encoding="latin-1")
    data.columns = data.columns.str.strip()

    t_data = data.iloc[:, 0].to_numpy(dtype=float)
    T_data = 9.81 * data.iloc[:, 1].to_numpy(dtype=float)

    # Evitar negativos
    T_data = np.maximum(T_data, 0.0)

    # =========================
    # 🔧 DETECTAR INICIO REAL DE EMPUJE
    # =========================
    threshold = 3.0  # Newton (ajustable)

    idx_start = None
    for i in range(len(T_data)):
        if T_data[i] > threshold:
            idx_start = i
            break

    if idx_start is None:
        raise ValueError(f"{filename}: no se detectó empuje válido")

    # Recortar desde inicio real
    t_data = t_data[idx_start:]
    T_data = T_data[idx_start:]

    # =========================
    # 🔧 REAJUSTAR TIEMPO A 0
    # =========================
    t_data = t_data - t_data[0]

    # =========================
    # INTERPOLADOR
    # =========================
    T_interp = interp1d(
        t_data,
        T_data,
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    burn_time = t_data[-1]

    return T_interp, burn_time


def rocket_ode(t, y, rocket, thrust_fun, burn_time, launch_angle_rad, use_aero=True):
    x, z, vx, vz, m = y

    z_eff = max(z, 0.0)

    V = np.sqrt(vx**2 + vz**2)
    thrust = float(thrust_fun(t))

    if t <= burn_time:
        mdot = -(rocket.m0 - rocket.mf) / burn_time
    else:
        mdot = 0.0

    theta_T = launch_angle_rad

    Fx = thrust * np.cos(theta_T)
    Fz = thrust * np.sin(theta_T)

    if use_aero and V > 1e-6:
        _, _, rho = atmosphere_isa(z_eff)
        q_dyn = 0.5 * rho * V**2
        gamma = np.arctan2(vz, vx)

        if rocket.Cd is not None:
            D = q_dyn * rocket.area_ref * rocket.Cd
            Fx += -D * np.cos(gamma)
            Fz += -D * np.sin(gamma)

        if rocket.Cl is not None:
            L = q_dyn * rocket.area_ref * rocket.Cl
            Fx += -L * np.sin(gamma)
            Fz +=  L * np.cos(gamma)

    g = gravity(z_eff)

    dxdt = vx
    dzdt = vz
    dvxdt = Fx / m
    dvzdt = Fz / m - g
    dmdt = mdot

    return [dxdt, dzdt, dvxdt, dvzdt, dmdt]


def simulate_rocket(
    rocket,
    thrust_excel,
    launch_angle_deg,
    time_col="t",
    thrust_col="T",
    t_final=45.0,
    V0=0.0,
    use_aero=True
):
    thrust_fun, burn_time = load_thrust_curve_excel(
        thrust_excel,
        time_col=time_col,
        thrust_col=thrust_col
    )

    theta0 = np.deg2rad(launch_angle_deg)

    y0 = [
        0.0,
        0.5,
        V0 * np.cos(theta0),
        V0 * np.sin(theta0),
        rocket.m0
    ]

    def impact_event(t, y):
        z = y[1]
        vz = y[3]

        if t < 1.0:
            return 1.0

        if vz >= 0:
            return 1.0

        return z

    impact_event.terminal = True
    impact_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, y: rocket_ode(
            t, y, rocket, thrust_fun, burn_time, theta0, use_aero
        ),
        t_span=(0.0, t_final),
        y0=y0,
        max_step=0.001,
        events=impact_event,
        rtol=1e-7,
        atol=1e-9
    )

    x = sol.y[0]
    z = sol.y[1]
    vx = sol.y[2]
    vz = sol.y[3]
    v = np.sqrt(vx**2 + vz**2)

    print("\nSimulación")
    print(f"Archivo:             {thrust_excel}")
    print(f"Tiempo final:        {sol.t[-1]:.6f} s")
    print(f"Altitud máxima:      {np.max(z):.3f} m")
    print(f"Alcance máximo:      {np.max(x):.3f} m")
    print(f"Vel. máxima:         {np.max(v):.3f} m/s")

    return sol


# =========================
# DEFINICIÓN DEL COHETE
# =========================
rocket = Rocket(
    m0=8.7,
    mf=7.3,
    diameter=0.103,
    Cd=0.6,
    Cl=None
)

# =========================
# LISTA DE ARCHIVOS
# =========================
archivos = [
    "load_cell_1.csv",
    "load_cell_2.csv",
    "load_cell_3.csv",
    "load_cell_4.csv",
    "load_cell_5.csv",
    "load_cell_6.csv",
]

# =========================
# LOOP PRINCIPAL
# =========================
for archivo in archivos:

    sol = simulate_rocket(
        rocket=rocket,
        thrust_excel=archivo,
        launch_angle_deg=85,
        time_col="Tiempo",
        thrust_col="Fuerza",
        t_final=80,
        V0=0.0,
        use_aero=True
    )

    t = sol.t
    x = sol.y[0]
    z = sol.y[1]
    vx = sol.y[2]
    vz = sol.y[3]
    v = np.sqrt(vx**2 + vz**2)

    # Métricas
    z_max = np.max(z)
    x_max = np.max(x)
    v_max = np.max(v)

    # Eventos
    idx_apogeo = np.argmax(z)
    t_apogeo = t[idx_apogeo]
    z_apogeo = z[idx_apogeo]

    z_target = z_apogeo - 5.0

    idx_5m = None
    for i in range(idx_apogeo, len(z)):
        if z[i] <= z_target:
            idx_5m = i
            break

    t_5m = t[idx_5m] if idx_5m is not None else None
    t_3s = t_5m + 3.0 if t_5m is not None else None

    # =========================
    # GRÁFICO
    # =========================
    fig, ax1 = plt.subplots()

    ax1.plot(t, z, label="Altitud")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Altitud [m]")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(t, v, linestyle="--", label="Velocidad")
    ax2.set_ylabel("Velocidad [m/s]")

    # Líneas
    ax1.axvline(t_apogeo, linestyle="--")
    ax1.text(t_apogeo, z_apogeo, "Apogeo", rotation=90, va="bottom")

    if t_5m is not None:
        ax1.axvline(t_5m, linestyle="--")
        ax1.text(t_5m, z_target, "-5 m", rotation=90, va="bottom")

    if t_3s is not None:
        ax1.axvline(t_3s, linestyle="--")
        ax1.text(t_3s, 0.1*z_apogeo, "+3 s", rotation=90, va="bottom")

    # Leyenda con métricas
    texto_leyenda = (
        f"Altura máx: {z_max:.1f} m\n"
        f"Alcance máx: {x_max:.1f} m\n"
        f"Vel máx: {v_max:.1f} m/s"
    )

    ax1.legend(loc="upper left", title=texto_leyenda)

    plt.title(f"Resultado: {archivo}")
    plt.show()

        # =========================
    # GRÁFICO EXTRA:
    # ALTURA VS ALCANCE
    # =========================
    plt.figure()

    plt.plot(x, z)

    plt.xlabel("Alcance [m]")
    plt.ylabel("Altitud [m]")
    plt.title(f"Altura vs Alcance: {archivo}")

    plt.grid()

    plt.show()