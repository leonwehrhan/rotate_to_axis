import numpy as np
import mdtraj as md


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
