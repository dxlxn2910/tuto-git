# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 07:18:26 2025

@author: HP
"""


import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import math
import torch, torch.nn as nn, torch.nn.functional as F

# Configuration du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== DÉFINITION DES PARAMÈTRES PHYSIQUES ==========
sigma = 5.67e-8  # Constante de Stefan-Boltzmann [W/m²K⁴]
lf = 0.114      # Épaisseur du lit de combustible [m]
L_fl = 1.69     # Longueur de la flamme [m]
Uw = 1.1        # Vitesse du vent ambiant [m/s]
Omega_s_deg = 17  # Angle de pente du lit de combustible [°]
Omega_s = math.radians(Omega_s_deg)
rho_f = 609     # Masse volumique du combustible [kg/m³]
phi = 0.008     # Rapport de compactage [-]
cp_f = 2500     # Capacité thermique massique du combustible [J/kgK]
h_vap = 2.25e6   # Enthalpie de vaporisation de l'eau [J/kg]
T_ig = 561.0     #Température d'ignition [K]
T_inf = 303.0   #Température ambiante [K]
Mw_inf=0.11
T_fl = 1083.0   # Température de la flamme [K]
T_b = 561.0     #Température des braises [K]
D = 0.00252     #Diamètre des particules de combustible [m]
w = 0.686       #Largeur du lit de combustible [m]
afb = 0.6       #Absorptivité du lit de combustible [-]
eps_fb = 0.9    #Émissivité du lit de combustible [-]
s = 17.5        #Surface spécifique des particules de combustible [m⁻¹]
k_fl = 0.1      #Conductivité thermique de la flamme [W/mK]
k_b = 0.0495
#mu = 1.8e-5     #Viscosité dynamique de l'air [Pa·s]
mu = 6.2e-5     #Viscosité dynamique de l'air à 400 degré [Pa·s]
Pr = 0.71       #Nombre de Prandtl [-]


# === Fonctions intermediares  ===
def Omega_w(Uw=Uw, g=9.81, L_fl=L_fl):
    Uw_t = torch.tensor(Uw)
    gL_t = torch.tensor(g * L_fl)
    return torch.atan(1.4 * Uw_t / torch.sqrt(gL_t))

def theta(Uw=Uw, Omega_s=Omega_s):
    return torch.tensor(Omega_s) + Omega_w(Uw)

def Z(y, L_fl=L_fl):
    th = theta()
    return (y / L_fl - torch.sin(th)) / torch.cos(th)

# === definition des flux ===
def epsilon_fl(L_fl=L_fl):
    return 1 - torch.exp(torch.tensor(-0.6 * L_fl))

def E_fl():  # Flux émis par la flamme
    return epsilon_fl() * sigma * T_fl**4

def E_b():
    return sigma * T_b**4

def q_rs(y):  # Chaleur par rayonnement de la flamme
    Z_val = Z(y)
    tanh_term = torch.tanh(torch.tensor((2 / 3) * (w / L_fl)**(1 / 3)))
    return (afb * E_fl() / (2 * lf)) * (1 - Z_val / torch.sqrt(1 + Z_val**2)) * tanh_term

def q_ri(y):  # Rayonnement du lit de braises
    return 0.25 * s * E_b() * torch.exp(-0.25 * s * y)

def q_pr(y, T):  # Rayonnement des braises
    return -eps_fb * sigma / lf * (T**4 - T_inf**4) #expression non linearisé

#def q_pr(y, T):
#   T_ref = 373.0  # Température de référence pour la linéarisation du rayonnement autour de T_ref
#   return -eps_fb * sigma / lf * (4 * T_ref**3 * T - 3 * T_ref**4 - T_inf**4) #expression linéarisé. 

def Re_y(y):  # Nombre de Reynolds local
    return rho_f * Uw * y / mu

def q_cs(y, T):
    eps = 1e-1  # pour éviter la division par zéro
    y_safe = y + eps
    Re = Re_y(y_safe)
    coeff = 0.565 * k_fl * Re.sqrt() * Pr**0.5
    base = y_safe * lf
    exponent = torch.exp(-0.3 * y_safe / L_fl)
    q_val = coeff / base * (T_fl - T) * exponent
    return q_val  # bornes amples mais raisonnables


def Re_D():
    U_fb = (1 - phi) * Uw
    return rho_f * U_fb * D / mu

def q_ci(y, T):  # Convection du lit de braises
    Re = Re_D()
    return (0.911 * s * k_b * Re**0.385 * Pr**(1 / 3)) / D * (T_b - T) * torch.exp(-0.25 * s * y)

def Q_1(y, T):
    q = q_rs(y) + q_ri(y) + q_pr(y, T) + q_cs(y, T) + q_ci(y, T)
    return q

def Q_2(y):
    """Calcule Q_2(y) en fixant T=373K dans toutes les fonctions."""
    T_fixed = torch.tensor(373.0).to(y.device)
    
    # Appel des fonctions originales avec T=373K imposé
    q_rs_val = q_rs(y)                           # Ne dépend pas de T
    q_ri_val = q_ri(y)                           # Ne dépend pas de T
    q_pr_val = q_pr(y, T_fixed)                  # T forcé à 373K
    q_cs_val = q_cs(y, T_fixed)                  # T forcé à 373K
    q_ci_val = q_ci(y, T_fixed)                  # T forcé à 373K
    
    return q_rs_val + q_ri_val + q_pr_val + q_cs_val + q_ci_val

# ------------------------------------------------------------------
# Réseau de base (sigmoïde pour borner la sortie, out_scale réglable)
# ------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, out_scale=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.out_scale = out_scale

    def forward(self, y):
        return self.out_scale * torch.sigmoid(self.net(y))


# ------------------------------------------------------------------
# Modèle conjoint : T(y), Mw(y) + paramètre R appris
# ------------------------------------------------------------------
class JointModel(nn.Module):
    def __init__(self, lambda_chi=1000.0, R_min=0.0, R_max=0.08):
        super().__init__()
        self.T_net  = Net(out_scale=300.0)
        self.Mw_net = Net(out_scale=0.11)
        self.R_raw  = nn.Parameter(torch.tensor(0.0))
        self.lambda_chi = lambda_chi
        self.R_min = R_min
        self.R_max = R_max

    def forward(self, y):
        T  = 300.0 + self.T_net(y)
        Mw = self.Mw_net(y)
        return T, Mw

    def R(self):
        return self.R_min + (self.R_max - self.R_min) * torch.sigmoid(self.R_raw)

    def chi(self, T):
        return torch.sigmoid(self.lambda_chi * (T - 373.0))

# ------------------------------------------------------------------
# Boucle d’entraînement
# ------------------------------------------------------------------
def train_joint_model(n_epochs=10000, lambda_chi=10000.0, lr=1e-3):
    model     = JointModel(lambda_chi=lambda_chi).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # points de collocation
    y_train = torch.linspace(0, 12, 120, device=device).view(-1, 1)
    y_train.requires_grad_()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        T, Mw = model(y_train)
        R     = model.R()
        chi   = model.chi(T)

        # dérivées
        dT_dy  = autograd.grad(T,  y_train, torch.ones_like(T),  create_graph=True)[0]
        dMw_dy = autograd.grad(Mw, y_train, torch.ones_like(Mw), create_graph=True)[0]

        # résidus pondérés
        res_T  = (1 - chi) * (dT_dy + Q_1(y_train, T) / (rho_f * cp_f  * phi * R))
        res_Mw = chi       * (dMw_dy - Q_2(y_train)   / (rho_f * h_vap * phi * R))

        loss_pde = torch.mean(res_T**2 + res_Mw**2)

        # condition initiale (Cauchy) en y = 12  (T= T_inf, Mw = Mw_inf)
        y_ic = torch.tensor([[12.0]], device=device)
        T_ic, Mw_ic = model(y_ic)
        loss_ic = (9_0000 * (T_ic - T_inf)**2 + 1e20 * (Mw_ic - Mw_inf)**2)

        loss = loss_pde + loss_ic
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | loss={loss.item():.3e} | R={R.item():.4f}")

    return model

# ------------------------------------------------------------------
# Entraînement + diagnostics rapides
# ------------------------------------------------------------------
model = train_joint_model(n_epochs=5000, lambda_chi=100)

with torch.no_grad():
    R_learned = model.R().item()
    print(f"\nParamètre R appris : {R_learned:.4f}")

    # profils fins
    y_fine = torch.linspace(0, 12, 600, device=device).view(-1, 1)
    T_val, Mw_val = model(y_fine)
    y_np  = y_fine.cpu().numpy().flatten()
    T_np  = T_val.cpu().numpy().flatten()
    Mw_np = Mw_val.cpu().numpy().flatten()

# === Analyse post-entraînement pour extraire les valeurs ===
with torch.no_grad():
    y_fine = torch.linspace(0, 12, 1000, device=device).view(-1, 1)
    T_vals, Mw_vals = model(y_fine)

    target_temp = 373.0
    diffs = torch.abs(T_vals - target_temp)
    min_diff, idx = torch.min(diffs, dim=0)
    y_target = y_fine[idx].item()
    
    target_Mw = 0.11
    diffs_Mw = torch.abs(Mw_vals - target_Mw)
    min_diff_Mw, idx_Mw = torch.min(diffs_Mw, dim=0)
    y_Mw_target = y_fine[idx_Mw].item()

    T0 = model(torch.tensor([[0.0]], device=device))[0].item()
    T12 = model(torch.tensor([[12.0]], device=device))[0].item()
    Mw0 = model(torch.tensor([[0.0]], device=device))[1].item()
    Mw12 = model(torch.tensor([[12.0]], device=device))[1].item()

# === Tracé des résultats avec y_target ===
def plot_results(model, y_target, y_Mw_target):
    y = torch.linspace(0, 12, 300, device=device).view(-1, 1)
    T, Mw = model(y)
    y_np = y.cpu().detach().numpy()
    T_np = T.cpu().detach().numpy()
    Mw_np = Mw.cpu().detach().numpy()

    plt.figure(figsize=(10, 4))

    # === Température ===
    plt.subplot(1, 2, 1)
    plt.plot(y_np, T_np, label="Température T(y)", color='orange')
    plt.axhline(T_inf, color='red', linestyle='--', label="T(12) attendu")
    plt.axhline(373, color='gray', linestyle=':', linewidth=1.2, label="Seuil T = 373 K")
    plt.scatter([y_target], [373], color='green', marker='X', s=100, label=f"T=373K à y={y_target:.2f} m")
    plt.axvline(x=y_target, color='green', linestyle=':', linewidth=1.2)
    plt.xlabel("y (m)"); plt.ylabel("T (K)")
    plt.title("Profil de Température"); plt.grid(); plt.legend()

    # === Humidité ===
    plt.subplot(1, 2, 2)
    plt.plot(y_np, Mw_np, label="Humidité Mw(y)", color='blue')
    plt.axhline(Mw_inf, color='red', linestyle='--', label="Mw(12) attendue")
    plt.scatter([y_Mw_target], [0.11], color='purple', marker='X', s=100, label=f"Mw=0.11 à y={y_Mw_target:.2f} m")
    plt.axvline(x=y_Mw_target, color='purple', linestyle=':', linewidth=1.2)
    plt.xlabel("y (m)"); plt.ylabel("Mw")
    plt.title("Profil d’Humidité"); plt.grid(); plt.legend()

    plt.tight_layout()
    plt.show()



plot_results(model, y_target, y_Mw_target)

#==================graphe combinée========================================================

def plot_combined_graph(model, y_target):
    y = torch.linspace(0, 12, 300, device=device).view(-1, 1)
    T, Mw = model(y)
    y = y.cpu().detach().numpy().flatten()
    T = T.cpu().detach().numpy().flatten()
    Mw = Mw.cpu().detach().numpy().flatten()

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Axe 1 : Température
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("Température T(y) [K]", color='orange')
    ax1.plot(y, T, color='orange', label="Température T(y)")
    ax1.axhline(373, color='gray', linestyle=':', linewidth=1.2, label="Seuil T = 373 K")
    ax1.axhline(T_inf, color='red', linestyle='--', label="T(12) attendu")
    ax1.axvline(x=y_target, color='green', linestyle=':', linewidth=1.2)
    ax1.scatter([y_target], [373], color='green', marker='X', s=100, label=f"T=373K à y={y_target:.2f} m")
    ax1.tick_params(axis='y', labelcolor='orange')

    # Axe 2 : Mw
    ax2 = ax1.twinx()
    ax2.set_ylabel("Humidité Mw(y)", color='blue')
    ax2.plot(y, Mw, color='blue', label="Humidité Mw(y)")
    ax2.axhline(Mw_inf, color='red', linestyle='--', label="Mw(12) attendue")
    ax2.tick_params(axis='y', labelcolor='blue')

    # Légende combinée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    fig.tight_layout()
    plt.title("Profils combinés Température & Humidité")
    plt.grid()
    plt.show()
    
plot_combined_graph(model, y_target)


# === Résumé imprimé ===
print("\n=== Résumé des résultats ===")
#print(f"Valeur de R utilisée : {R}")
print(f"y* tel que T(y*) ≈ 373 K : y = {y_target:.4f} m")
print(f"y* tel que Mw(y*) ≈ 0.11 : y = {y_Mw_target:.4f} m")
print(f"T(0)  = {T0:.2f} K")
print(f"T(12) = {T12:.2f} K")
print(f"Mw(0)  = {Mw0:.5f}")
print(f"Mw(12) = {Mw12:.5f}")


# ==== Paramètres ====
l_f = 0.114  # Épaisseur du lit de combustible [m]
y = torch.linspace(0, 12, 100).view(-1, 1).to(device)

# Récupération des profils T et Mw
with torch.no_grad():
    T_pred, Mw_pred = model(y)
    T = T_pred.detach()
    Mw = Mw_pred.detach()

# Calcul des flux volumiques (en kW/m³)
q_rs_vol = q_rs(y).detach().cpu().numpy().flatten() / l_f * 0.00016
q_ri_vol = q_ri(y).detach().cpu().numpy().flatten() / l_f * 0.00015
q_cs_vol = q_cs(y, T).detach().cpu().numpy().flatten() / l_f * 0.0000047
q_pr_vol = q_pr(y, T).detach().cpu().numpy().flatten() / l_f * 0.000099
q_ci_vol = q_ci(y, T).detach().cpu().numpy().flatten() / l_f * 0.0000008

# ==== Tracé des contributions ====
plt.figure(figsize=(10, 6))
plt.plot(y.cpu().numpy(), q_rs_vol, label='Radiation surfacique', linestyle='--', color='red')
plt.plot(y.cpu().numpy(), q_cs_vol, label='Convection surfacique', linestyle='-.', color='blue')
plt.plot(y.cpu().numpy(), q_ri_vol, label='Radiation interne', linestyle=':', color='green')
plt.plot(y.cpu().numpy(), q_ci_vol, label='Convection interne', color='purple')
plt.plot(y.cpu().numpy(), q_pr_vol, label='Perte radiative', linestyle='--', color='orange')

plt.xlabel('Distance à la flamme y [m]')
plt.ylabel('Contribution des flux [kW/m³]')
plt.title('Contribution des flux vs Distance (Bouleau blanc, +17° pente, 1.1 m/s vent)')
plt.legend()
plt.grid(True)
plt.xlim(0, 12)
plt.ylim(-60, 150)
plt.tight_layout()
plt.show()


y_values = torch.tensor([[0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]], device=device).T
T_values, Mw_values = model(y_values)

print("  y (m) | T(y) (K) | Mw(y) |  q_rs  |  q_cs  |  q_ri  |  q_ci  |  q_pr   | Total")
print("--------------------------------------------------------------------------------")
for y_i, T_i, Mw_i in zip(y_values, T_values, Mw_values):
    q_rs_val = q_rs(y_i).item() / l_f * 0.00016
    q_cs_val = q_cs(y_i, T_i).item() / l_f * 0.0000047
    q_ri_val = q_ri(y_i).item() / l_f * 0.00015
    q_ci_val = q_ci(y_i, T_i).item() / l_f *  0.00000099
    q_pr_val = q_pr(y_i, T_i).item() / l_f * 0.000099
    total = q_rs_val + q_cs_val + q_ri_val + q_ci_val + q_pr_val
    print(f"{y_i.item():6.1f} | {T_i.item():7.1f} | {Mw_i.item():5.3f} | {q_rs_val:6.1f} | {q_cs_val:6.1f} | {q_ri_val:6.1f} | {q_ci_val:6.1f} | {q_pr_val:7.1f} | {total:6.1f}")