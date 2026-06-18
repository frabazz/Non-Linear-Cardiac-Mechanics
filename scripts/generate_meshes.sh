#!/usr/bin/env bash
# Generate .msh meshes from .geo files with gmsh (3D tetrahedra, msh2 format):
# geometry/beam/*.geo -> mesh/beam/*.msh, geometry/ventricle/*.geo -> mesh/ventricular_meshes/*.msh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GEO_ROOT="$PROJECT_ROOT/geometry"
MESH_ROOT="$PROJECT_ROOT/mesh"

if ! command -v gmsh >/dev/null 2>&1; then
  echo "error: 'gmsh' not found in PATH." >&2
  exit 1
fi

GMSH_OPTS=(-3 -format msh2)

convert_dir() {
  local in_subdir="$1"
  local out_subdir="$2"
  local in_dir="$GEO_ROOT/$in_subdir"
  local out_dir="$MESH_ROOT/$out_subdir"

  if [[ ! -d "$in_dir" ]]; then
    echo "  skipping '$in_subdir': directory '$in_dir' not found"
    return 0
  fi

  mkdir -p "$out_dir"

  shopt -s nullglob
  local geo_files=("$in_dir"/*.geo)
  shopt -u nullglob

  if [[ ${#geo_files[@]} -eq 0 ]]; then
    echo "  no .geo found in '$in_dir'"
    return 0
  fi

  local geo name msh
  for geo in "${geo_files[@]}"; do
    name="$(basename "$geo" .geo)"
    msh="$out_dir/$name.msh"

    if [[ -f "$msh" && "$msh" -nt "$geo" ]]; then
      echo "  [skip] $in_subdir/$name.geo"
      continue
    fi

    echo "  [mesh] $in_subdir/$name.geo -> $out_subdir/$name.msh"
    gmsh "$geo" "${GMSH_OPTS[@]}" -o "$msh"
  done
}

echo "generating meshes in '$MESH_ROOT'"
convert_dir "beam" "beam"
convert_dir "ventricle" "ventricular_meshes"
echo "done."
