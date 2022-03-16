import numpy as np
import mdtraj as md
import argparse


def rotate_structure(t, ang, rot_axis):
    '''
    Rotate molecular structure around rotation axis to given angle.

    Parameters
    ----------
    t : md.Trajectory
        Molecular structure as mdtraj trajectory with one frame.
    ang : float
        Angle in radians.
    rot_axis : np.ndarray
        Vector of rotation axis
    '''
    if t.n_frames != 1:
        raise ValueError(f't has {t.n_frames} frames, must have 1.')

    # normalize rotation axis
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # Define Parameters based on Euler-Rodriguez Formula
    a = np.cos(ang / 2)
    b = rot_axis[0] * np.sin(ang / 2)
    c = rot_axis[1] * np.sin(ang / 2)
    d = rot_axis[2] * np.sin(ang / 2)

    # Construct Rotation Matrix
    R = np.array([
        [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - c**2 - b**2]
    ])

    # Create new trajectory with rotated coordinates
    t_new = t
    for i in range(t_new.n_atoms):
        t_new.xyz[0][i] = np.dot(R, t_new.xyz[0][i])

    return t_new


def rotate_to_axis(t, vec, axis='z'):
    '''
    Rotate molecular structure so that defined vector aligns with one axis of the coordinate system.

    Parameters
    ----------
    t : md.Trajectory
        Molecular structure as mdtraj trajectory with one frame.
    vec : np.ndarray or list of int or list of str
        Vector that is aligned with coordinate axis. Vector can be defined as np.ndarray.
        Alternatlively, a list of two atom indices can be given. The vector between the
        atoms will be used. The list can also hold two strings, which will be used as selection
        strings to define two substructures. The vector will be between the centers of
        mass of the two substructures.
    axis : "x", "y" or "z"
        Coordinate axis to align on.
    '''
    if t.n_frames != 1:
        raise ValueError(f't has {t.n_frames} frames, must have 1.')

    # Verify that vec is np.ndarray or list with two int or str elements
    if type(vec) == np.ndarray:
        if vec.shape != (3,):
            raise ValueError(f'vec is shape {vec.shape}, must be shape (3,)')
    if type(vec) == list:
        if len(vec) != 2:
            raise ValueError(f'vec has {len(vec)} elements, must have two.')
        if type(vec[0]) != type(vec[1]):
            raise ValueError('vec holds two elements of different type')
        if type(vec[0]) != int and type(vec[0]) != str:
            raise ValueError('Elements in vec must be int or str')
    else:
        raise ValueError(f'vec is {type(vec)}, must be np.ndarray or list')

    # Get coordinate axis, raise error if not x, y or z
    if axis == 'x':
        axis_coord = np.array([1, 0, 0])
    elif axis == 'y':
        axis_coord = np.array([0, 1, 0])
    elif axis == 'z':
        axis_coord = np.array([0, 0, 1])
    else:
        raise ValueError(f'Unknown coordinate axis {axis}')

    # Define vector for alignment
    if type(vec) == np.ndarray:
        vec_al = vec
    elif type(vec[0]) == int:
        vec_al = t.xyz[0][vec[1]] - t.xyz[0][vec[0]]
    else:
        # Get atom indices of substrucures from selection
        idx_1 = t.top.select(vec[0])
        idx_2 = t.top.select(vec[1])

        # Extract substructures
        t_1 = t.atom_slice(idx_1)
        t_2 = t.atom_slice(idx_2)

        # Compute centers of mass
        com_1 = md.compute_center_of_mass(t_1)[0]
        com_2 = md.compute_center_of_mass(t_2)[0]

        # alignment vector
        vec_al = com_1 - com_2

    # Normalize alignment vector
    vec_al = vec_al / np.linalg.norm(vec_al)

    # Compute rotation axis
    rot_axis = np.cross(vec_al, axis_coord)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)

    # Compute rotation angle
    ang = np.arccos(np.dot(vec_al, axis_coord))

    # Rotate structure
    t_new = rotate_structure(t, ang, rot_axis)

    return t_new


if __name__ == '__main__':

    # Get input file argument
    parser = argparse.ArgumentParser()
    parser.add_argument('f', help='input structure file')
    args = parser.parse_args()
    f = args.f

    t = md.load(f)
