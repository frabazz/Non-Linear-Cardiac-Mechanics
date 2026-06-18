import os
import glob
import re
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd



MAX_MARKERS_PER_CURVE = 15

def compute_markevery(x, max_markers=MAX_MARKERS_PER_CURVE):
    n = len(x)
    if n <= 1:
        return 1
    return max(1, int(np.ceil(n / max_markers)))

d = 2.91
xi_in = 0.6
xi_out = 1.02
z_cut = 1.19

# Offset inward per campionamento su superfici (evita punti sul bordo)
XI_OFFSET = 1e-3

# Calcolo dei semiassi di riferimento
R_in_xy = d * np.sinh(xi_in)
R_in_z  = d * np.cosh(xi_in)
R_out_z = d * np.cosh(xi_out)

# cap integral area
r_cap_sq     = (R_in_xy**2) * (1 - (z_cut / R_in_z)**2)
A_cap        = np.pi * r_cap_sq
cap_integral = z_cut * A_cap

# apical reference points
endo_apex_ref = np.array([[0.0, 0.0, -R_in_z]])
epi_apex_ref  = np.array([[0.0, 0.0, -R_out_z]])

# endocardium reference line


mu_base_endo = np.arccos(z_cut / R_in_z)
mu_vals_endo = np.linspace(np.pi, mu_base_endo, 100)
endo_line_ref = np.column_stack((
    d * np.sinh(xi_in) * np.sin(mu_vals_endo),
    np.zeros_like(mu_vals_endo),
    d * np.cosh(xi_in) * np.cos(mu_vals_endo)
))

xi_endo_sample   = xi_in + XI_OFFSET
mu_base_endo_s   = np.arccos(z_cut / (d * np.cosh(xi_endo_sample)))
mu_vals_endo_s   = np.linspace(np.pi, mu_base_endo_s, 100)
endo_line_sample = np.column_stack((
    d * np.sinh(xi_endo_sample) * np.sin(mu_vals_endo_s),
    np.zeros_like(mu_vals_endo_s),
    d * np.cosh(xi_endo_sample) * np.cos(mu_vals_endo_s)
))

#
# midwall reference line

xi_mid    = (xi_in + xi_out) / 2.0
R_mid_z   = d * np.cosh(xi_mid)
mu_base   = np.arccos(z_cut / R_mid_z)
mu_vals   = np.linspace(np.pi, mu_base, 100)
mid_line_ref = np.column_stack((
    d * np.sinh(xi_mid) * np.sin(mu_vals),
    np.zeros_like(mu_vals),
    d * np.cosh(xi_mid) * np.cos(mu_vals)
))

#epicardio ref line
# 
mu_base_epi = np.arccos(z_cut / R_out_z)
mu_vals_epi = np.linspace(np.pi, mu_base_epi, 100)
epi_line_ref = np.column_stack((
    d * np.sinh(xi_out) * np.sin(mu_vals_epi),
    np.zeros_like(mu_vals_epi),
    d * np.cosh(xi_out) * np.cos(mu_vals_epi)
))

xi_epi_sample   = xi_out - XI_OFFSET
mu_base_epi_s   = np.arccos(z_cut / (d * np.cosh(xi_epi_sample)))
mu_vals_epi_s   = np.linspace(np.pi, mu_base_epi_s, 100)
epi_line_sample = np.column_stack((
    d * np.sinh(xi_epi_sample) * np.sin(mu_vals_epi_s),
    np.zeros_like(mu_vals_epi_s),
    d * np.cosh(xi_epi_sample) * np.cos(mu_vals_epi_s)
))


results    = {}
csv_data   = []
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)
output_dir = results_dir

folders = glob.glob(os.path.join(script_dir, "solution_*"))
if not folders:
    print("no folder found.")
    exit()

def get_step(filename):
    match = re.search(r'(\d+)\.p?vtu$', filename)
    return int(match.group(1)) if match else -1

def safe_sample(points_geom, points_sample, mesh, label, mesh_name):
    """
    samples solution on points_sample (slightly inside),
    then applies the displacement to the real geometric points (points_geom).
    Returns the deformed line, or None if sampling fails.
    """
    sampled = pv.PolyData(points_sample).sample(mesh)
    if 'solution' not in sampled.array_names:
        print(f"  Warning: 'solution' not found for {label} in {mesh_name}")
        return None
    displ = sampled['solution']
    if not np.any(np.abs(displ) > 1e-12):
        print(f"  Warning: zero displacements for {label} in {mesh_name} "
              f"(points probably outside the mesh)")
        return None
    return points_geom + displ

for folder in folders:
    mesh_name = os.path.basename(folder).replace("solution_", "")
    files = glob.glob(os.path.join(folder, "output-*.pvtu")) or \
            glob.glob(os.path.join(folder, "output-*.vtu"))
    files.sort(key=get_step)
    if not files:
        continue

    print(f"Processing: {mesh_name} ({len(files)} steps)")
    steps, endo_z, epi_z, volumes, ventricle_volumes = [], [], [], [], []
    last_mid_line_def  = None
    last_endo_line_def = None
    last_epi_line_def  = None

    for f in files:
        step = get_step(f)
        if step == -1:
            continue

        try:
            mesh = pv.read(f)
            
        except Exception as e:
            print(f"  Error reading {f}: {e}")
            continue

        steps.append(step)

        # A. Apex Displacement
        endo_z.append(endo_apex_ref[0, 2] +
                      pv.PolyData(endo_apex_ref).sample(mesh)['solution'][0][2])
        epi_z.append(epi_apex_ref[0, 2] +
                     pv.PolyData(epi_apex_ref).sample(mesh)['solution'][0][2])

        # B. cavity volume
        surf    = mesh.extract_surface(algorithm=None)
        centers = surf.cell_centers().points
        rad_sq  = ((centers[:, 0] / R_in_xy)**2 +
                   (centers[:, 1] / R_in_xy)**2 +
                   (centers[:, 2] / R_in_z)**2)
        endo_mask = (rad_sq < 1.5) & (centers[:, 2] < 1.185)
        endo_surf = surf.extract_cells(endo_mask)

        endo_surf.points += endo_surf['solution']
        endo_surf = (endo_surf.extract_surface(algorithm=None)
                              .compute_normals(cell_normals=True, point_normals=False))

        integral_endo  = np.sum(
            endo_surf.cell_centers().points[:, 2] *
            (-endo_surf['Normals'][:, 2]) *
            endo_surf.compute_cell_sizes()['Area']
        )
        cavity_volume = integral_endo + cap_integral
        volumes.append(cavity_volume)

        # C. Ventricle Volume
        warped_mesh = mesh.warp_by_vector('solution')
        ventricle_volumes.append(warped_mesh.volume)

        csv_data.append({
            'Mesh': mesh_name,
            'Step': step,
            'Cavity_Volume_cm3': cavity_volume,
            'Ventricle_Volume_cm3': warped_mesh.volume
        })

        # D.deformation lines
        if f == files[-1]:

            last_mid_line_def = (mid_line_ref +
                                 pv.PolyData(mid_line_ref).sample(mesh)['solution'])


            last_endo_line_def = safe_sample(
                endo_line_ref, endo_line_sample, mesh, "endocardio", mesh_name
            )


            last_epi_line_def = safe_sample(
                epi_line_ref, epi_line_sample, mesh, "epicardio", mesh_name
            )

    results[mesh_name] = {
        'steps':            steps,
        'endo_z':           endo_z,
        'epi_z':            epi_z,
        'volumes':          volumes,
        'ventricle_volumes': ventricle_volumes,
        'last_mid_line':    last_mid_line_def,
        'last_endo_line':   last_endo_line_def,
        'last_epi_line':    last_epi_line_def,
    }


# 3. Graph Generation

pd.DataFrame(csv_data).to_csv(
    os.path.join(output_dir, "Volumes_Data.csv"), index=False
)
colors = plt.cm.tab10.colors


PRESSURE_MAX_KPA = 15.0
_global_max_step = max((max(d['steps']) for d in results.values() if d.get('steps')), default=0)

def steps_to_pressure_kpa(steps, p_max_kpa=PRESSURE_MAX_KPA, max_step=_global_max_step):
    steps_arr = np.asarray(steps, dtype=float)
    if max_step <= 0:
        return steps_arr * 0.0
    return (steps_arr / float(max_step)) * float(p_max_kpa)

# Graph 1: Apex Displacement
plt.figure(figsize=(10, 6))
for i, (name, data) in enumerate(results.items()):
    c = colors[i % len(colors)]
    x_p = steps_to_pressure_kpa(data['steps'])
    me = compute_markevery(x_p)
    plt.plot(x_p, data['endo_z'], marker='o', markevery=me, color=c,
             label=f'{name} (Endo)')
    plt.plot(x_p, data['epi_z'], marker='s', markevery=me, linestyle='--', color=c,
             label=f'{name} (Epi)')
plt.axhline(y=-R_in_z,  color='gray',  linestyle=':', label='Endo Ref')
plt.axhline(y=-R_out_z, color='black', linestyle=':', label='Epi Ref')
plt.title("Apex Displacement")
plt.xlabel("Pressure (kPa)"); plt.ylabel("Z Position (cm)")
plt.xlim(0.0, PRESSURE_MAX_KPA)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Apex_Displacements.png")); plt.close()

# Graph 2: Side Expansion – Midwall
plt.figure(figsize=(8, 8))
plt.plot(mid_line_ref[:, 0], mid_line_ref[:, 2], 'k--', linewidth=2, label="Reference")
for i, (name, data) in enumerate(results.items()):
    if data['last_mid_line'] is not None:
        plt.plot(data['last_mid_line'][:, 0], data['last_mid_line'][:, 2],
                 label=name, color=colors[i % len(colors)])
plt.title("Side Expansion (Midwall)")
plt.xlabel("X (cm)"); plt.ylabel("Z (cm)")
plt.axis('equal'); plt.legend(); plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(output_dir, "Side_Expansion.png")); plt.close()

# Graph 2b: Side Expansion – Endocardio
plt.figure(figsize=(8, 8))
plt.plot(endo_line_ref[:, 0], endo_line_ref[:, 2], 'k--', linewidth=2, label="Reference")
for i, (name, data) in enumerate(results.items()):
    if data['last_endo_line'] is not None:
        plt.plot(data['last_endo_line'][:, 0], data['last_endo_line'][:, 2],
                 label=name, color=colors[i % len(colors)])
plt.title("Side Expansion (Endocardio)")
plt.xlabel("X (cm)"); plt.ylabel("Z (cm)")
plt.axis('equal'); plt.legend(); plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(output_dir, "Side_Expansion_Endocardio.png")); plt.close()

# Graph 2c: Side Expansion – Epicardio
plt.figure(figsize=(8, 8))
plt.plot(epi_line_ref[:, 0], epi_line_ref[:, 2], 'k--', linewidth=2, label="Reference")
for i, (name, data) in enumerate(results.items()):
    if data['last_epi_line'] is not None:
        plt.plot(data['last_epi_line'][:, 0], data['last_epi_line'][:, 2],
                 label=name, color=colors[i % len(colors)])
plt.title("Side Expansion (Epicardio)")
plt.xlabel("X (cm)"); plt.ylabel("Z (cm)")
plt.axis('equal'); plt.legend(); plt.grid(True, alpha=0.5)
plt.savefig(os.path.join(output_dir, "Side_Expansion_Epicardio.png")); plt.close()

# Graph 3: Cavity Volume
plt.figure(figsize=(10, 6))
for i, (name, data) in enumerate(results.items()):
    x_p = steps_to_pressure_kpa(data['steps'])
    me = compute_markevery(x_p)
    plt.plot(x_p, data['volumes'], marker='o', markevery=me,
             color=colors[i % len(colors)], label=name)
plt.title("Cavity Volume")
plt.xlabel("Pressure (kPa)"); plt.ylabel("Volume (cm³)")
plt.xlim(0.0, PRESSURE_MAX_KPA)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Cavity_Volume.png")); plt.close()

# Graph 4: Ventricle Volume
plt.figure(figsize=(10, 6))
for i, (name, data) in enumerate(results.items()):
    x_p = steps_to_pressure_kpa(data['steps'])
    me = compute_markevery(x_p)
    plt.plot(x_p, data['ventricle_volumes'], marker='o', markevery=me,
             color=colors[i % len(colors)], label=name)
plt.title("Ventricle Wall Volume")
plt.xlabel("Pressure (kPa)"); plt.ylabel("Volume (cm³)")
plt.xlim(0.0, PRESSURE_MAX_KPA)
plt.ylim(120, 130)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.5); plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Ventricle_Volume.png")); plt.close()

print("Successfull analysis. File generated in 'results'.")