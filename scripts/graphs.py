import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../build/solution_beam_02"))


FILE_VTU_FINE = os.path.join(SCRIPT_DIR, "output-beam_lollo_100.pvtu")
ARRAY_SPOSTAMENTO = "solution"

def get_deformed_positions(mesh, X_points):
    cloud = pv.PolyData(X_points)
    sampled = cloud.sample(mesh)
    disp = sampled[ARRAY_SPOSTAMENTO]
    return X_points + disp

try:
    mesh_fine = pv.read(FILE_VTU_FINE)
except Exception as e:
    print(f"Errore fatale: impossibile caricare {FILE_VTU_FINE}: {e}")
    exit()

x_vals = np.linspace(0, 10, 100)
line_points = np.column_stack((x_vals, np.full(100, 0.5), np.full(100, 0.5)))
def_line = get_deformed_positions(mesh_fine, line_points)

plt.figure(figsize=(10, 4))
plt.plot(def_line[:, 0], def_line[:, 2], 'r-', linewidth=2, label="La Tua Simulazione")
plt.title("Problem 1 - Figure 4: Deformation of a line")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

fig4_path = os.path.join(SCRIPT_DIR, "Fig4_LineDeformation.png")
plt.savefig(fig4_path, bbox_inches='tight')
print(f"Figura 4 salvata in: {fig4_path}")

#figure 5: Strain results
def calc_strain(mesh, X1, X2):
    x1 = get_deformed_positions(mesh, X1)
    x2 = get_deformed_positions(mesh, X2)
    dist_X = np.linalg.norm(X2 - X1, axis=1)
    dist_x = np.linalg.norm(x2 - x1, axis=1)
    return ((dist_x - dist_X) / dist_X) * 100.0

i_x = np.arange(0, 9)
X1_x = np.column_stack((i_x, np.full(9, 0.5), np.full(9, 0.5)))
X2_x = np.column_stack((i_x + 1.0, np.full(9, 0.5), np.full(9, 0.5)))

i_yz = np.arange(0, 10)
X1_y = np.column_stack((i_yz, np.full(10, 0.5), np.full(10, 0.5)))
X2_y = np.column_stack((i_yz, np.full(10, 0.9), np.full(10, 0.5)))

X1_z = np.column_stack((i_yz, np.full(10, 0.5), np.full(10, 0.5)))
X2_z = np.column_stack((i_yz, np.full(10, 0.5), np.full(10, 0.9)))

strain_x = calc_strain(mesh_fine, X1_x, X2_x)
strain_y = calc_strain(mesh_fine, X1_y, X2_y)
strain_z = calc_strain(mesh_fine, X1_z, X2_z)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels_x = [f"p{i+1}" for i in i_x]
labels_yz = [f"p{i+1}" for i in i_yz]

axes[0].plot(labels_x, strain_x, marker='o', color='tab:blue', label='La Tua Simulazione')
axes[0].set_title('x-axis strain (%)')
axes[0].set_ylabel('Strain (%)')
axes[0].grid(True, linestyle='--', alpha=0.7)

axes[1].plot(labels_yz, strain_y, marker='s', color='tab:orange')
axes[1].set_title('y-axis strain (%)')
axes[1].grid(True, linestyle='--', alpha=0.7)

axes[2].plot(labels_yz, strain_z, marker='^', color='tab:green')
axes[2].set_title('z-axis strain (%)')
axes[2].grid(True, linestyle='--', alpha=0.7)

fig.suptitle("Problem 1 - Figure 5: Strain Results", fontweight='bold')
plt.tight_layout()

fig5_path = os.path.join(SCRIPT_DIR, "Fig5_Strain.png")
plt.savefig(fig5_path, bbox_inches='tight')
print(f"Figura 5 salvata in: {fig5_path}")