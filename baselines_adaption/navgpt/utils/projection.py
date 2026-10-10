"""
Coordinate projection utilities for NavGPT ASAR adaptation.

Provides functions to back-project pixel coordinates from the camera frame
into the AirSim NED world frame.

Reused from ``baselines_adaption/semexp/utils/projection.py``.
"""

import numpy as np


def quaternion_to_rotation_matrix(quaternion):
    """Convert an AirSim quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    quaternion : airsim.Quaternionr (or duck-typed object with w/x/y/z_val)

    Returns
    -------
    R : ndarray (3, 3)
    """
    qw = quaternion.w_val
    qx = quaternion.x_val
    qy = quaternion.y_val
    qz = quaternion.z_val

    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,    2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def _yaw_rotation_matrix(yaw_deg):
    """Return a 3x3 NED rotation matrix for a pure yaw (in degrees).

    Transforms a vector from a yaw-rotated frame (camera) to the base frame
    (vehicle body).  Rotation is around the Z (down) axis.

    For yaw=90 (right camera), a forward-pointing vector [d,0,0] maps to
    the body-right direction [0,d,0].
    """
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])


def quaternion_rotate_vector(quaternion, vector_np):
    """Rotate a 3-D vector by an AirSim quaternion."""
    R = quaternion_to_rotation_matrix(quaternion)
    return R @ vector_np


def pixel_to_world(v, u, depth, image_shape, camera_fov,
                   camera_position, camera_orientation,
                   camera_yaw_offset=0.0):
    """Back-project a single pixel (u, v) with known depth to world NED coords.

    Parameters
    ----------
    v, u : int
        Row / column indices in the image.
    depth : float
        Metric depth at (v, u) in metres.
    image_shape : (H, W)
    camera_fov : float
        Horizontal field-of-view in degrees.
    camera_position : airsim.Vector3r
    camera_orientation : airsim.Quaternionr
    camera_yaw_offset : float
        Additional yaw of this camera relative to the vehicle body (degrees).
        0 = front, 90 = right, 180 = back, -90 = left.

    Returns
    -------
    dict  ``{"x": float, "y": float, "z": float}`` in AirSim NED metres.
    """
    height, width = image_shape
    fx = fy = width / (2 * np.tan(np.deg2rad(camera_fov / 2)))
    cx, cy = width / 2, height / 2

    # Normalised camera coords
    xn = (u - cx) / fx
    yn = (v - cy) / fy

    # Camera frame (OpenCV: Z forward, X right, Y down)
    z_cam = depth
    x_cam = xn * z_cam
    y_cam = yn * z_cam

    # Camera -> Body NED: body_X = cam_Z, body_Y = cam_X, body_Z = cam_Y
    point_ned = np.array([z_cam, x_cam, y_cam])

    # Apply camera yaw offset (rotate within the body frame)
    if camera_yaw_offset != 0.0:
        R_cam = _yaw_rotation_matrix(camera_yaw_offset)
        point_ned = R_cam @ point_ned

    # Rotate by vehicle orientation
    rotated = quaternion_rotate_vector(camera_orientation, point_ned)

    # Translate to world
    wx = rotated[0] + camera_position.x_val
    wy = rotated[1] + camera_position.y_val
    wz = rotated[2] + camera_position.z_val

    return {"x": float(wx), "y": float(wy), "z": float(wz)}
