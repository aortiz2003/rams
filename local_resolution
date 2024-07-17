import argparse
import mrcfile
import numpy as np
from scipy.spatial import KDTree

def parse_pdb_line(line):
    """Parse the relevant columns from a PDB line."""
    atom_type = line[12:16].strip()
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    b_value = float(line[60:66])
    return atom_type, x, y, z, b_value

def extract_and_modify_all_atoms(pdb_file, mrc_file, output_file, search_radius=5):
    try:
        # Read the PDB file
        with open(pdb_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        all_atoms = [parse_pdb_line(line) for line in lines if line.startswith("ATOM") or line.startswith("HETATM")]

        # Read the MRC file into RAM
        with mrcfile.open(mrc_file, permissive=True) as mrc:
            data = mrc.data.copy()  # Copy the data into RAM
            voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=float)
            origin = np.array([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=float)
            shape = data.shape

            # Adjust voxel size and origin if they are not defined in the MRC header
            if np.all(voxel_size == 0):
                voxel_size = np.array([1.0, 1.0, 1.0])
            if np.all(origin == 0):
                origin = np.array([0.0, 0.0, 0.0])

        def point_to_voxel(point):
            """Convert a 3D point to voxel coordinates."""
            return tuple(int((p - o) / v) for p, o, v in zip(point, origin, voxel_size))

        def voxel_to_point(voxel):
            """Convert voxel coordinates back to 3D point coordinates."""
            return tuple(v * vs + o for v, vs, o in zip(voxel, voxel_size, origin))

        # Create a KDTree for efficient neighbor search
        voxel_indices = np.array(np.nonzero(data)).T
        tree = KDTree(voxel_indices)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom = parse_pdb_line(line)
                    point = (atom[1], atom[2], atom[3])
                    voxel = point_to_voxel(point)

                    # Find all points within the search radius in the MRC data using KDTree
                    indices = tree.query_ball_point(voxel, search_radius)
                    closest_voxels = voxel_indices[indices]
                    closest_densities = data[tuple(closest_voxels.T)]
                    average_density = np.mean(closest_densities)
                    
                    # Modify the B-value in the original atom line
                    new_b_value_str = f"{average_density:6.2f}"
                    modified_line = line[:60] + new_b_value_str + line[66:]
                    outfile.write(modified_line)
                else:
                    outfile.write(line)
        print(f"Atom and HETATM lines with updated B-values successfully written to {output_file}.")
    except FileNotFoundError:
        print(f"Error: The file {pdb_file} or {mrc_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Modify B-values in a PDB file based on MRC density values within a given radius.")
    parser.add_argument("pdb_file", help="Path to the input PDB file")
    parser.add_argument("mrc_file", help="Path to the input MRC file")
    parser.add_argument("output_file", help="Path to the output PDB file")
    parser.add_argument("-r", "--radius", type=float, default=5, help="Search radius in angstroms (default: 5)")
    
    args = parser.parse_args()
    
    extract_and_modify_all_atoms(args.pdb_file, args.mrc_file, args.output_file, args.radius)

if __name__ == "__main__":
    main()
