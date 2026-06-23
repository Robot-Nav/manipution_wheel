from typing import Union
import torch
import numpy as np
import functools
from scipy.spatial.transform import Rotation


class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
            from_rep='axis_angle',
            to_rep='rotation_6d',
            from_convention=None,
            to_convention=None):
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        self.from_rep = from_rep
        self.to_rep = to_rep
        self.from_convention = from_convention
        self.to_convention = to_convention

    @staticmethod
    def _to_scipy_rep(rep, convention=None):
        """Map our rep names to scipy Rotation string names."""
        mapping = {
            'axis_angle': 'rotvec',
            'euler_angles': 'Euler',
            'quaternion': 'quat',
            'matrix': 'matrix',
        }
        return mapping[rep]

    @staticmethod
    def _from_matrix(R):
        """Rotation matrix to rotation_6d (Zhou et al.)."""
        return R[..., :2, :].reshape(*R.shape[:-2], 6)

    @staticmethod
    def _to_matrix(d6):
        """rotation_6d (Zhou et al.) to rotation matrix."""
        a1 = d6[..., :3]
        a2 = d6[..., 3:6]
        b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-9)
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-9)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack([b1, b2, b3], dim=-2)
        return R

    def _convert_to_matrix(self, x):
        """Convert from from_rep to rotation matrix using scipy."""
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.as_tensor(x, dtype=torch.float64)

        if self.from_rep == 'matrix':
            return x_t.float()

        if self.from_rep == 'rotation_6d':
            return self._to_matrix(x_t.float())

        if self.from_rep == 'axis_angle':
            rot = Rotation.from_rotvec(x_t.numpy())
        elif self.from_rep == 'euler_angles':
            rot = Rotation.from_euler(self.from_convention, x_t.numpy())
        elif self.from_rep == 'quaternion':
            rot = Rotation.from_quat(x_t.numpy())
        else:
            raise ValueError(f"Unsupported from_rep: {self.from_rep}")

        mat = torch.from_numpy(rot.as_matrix()).float()
        return mat

    def _convert_from_matrix(self, mat):
        """Convert from rotation matrix to to_rep using scipy."""
        if self.to_rep == 'matrix':
            return mat

        if self.to_rep == 'rotation_6d':
            return self._from_matrix(mat)

        mat_np = mat.double().numpy()
        rot = Rotation.from_matrix(mat_np)

        if self.to_rep == 'axis_angle':
            result = torch.from_numpy(rot.as_rotvec()).float()
        elif self.to_rep == 'euler_angles':
            result = torch.from_numpy(rot.as_euler(self.to_convention)).float()
        elif self.to_rep == 'quaternion':
            result = torch.from_numpy(rot.as_quat()).float()
        else:
            raise ValueError(f"Unsupported to_rep: {self.to_rep}")

        return result

    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        is_numpy = isinstance(x, np.ndarray)
        mat = self._convert_to_matrix(x)
        result = self._convert_from_matrix(mat)
        if is_numpy:
            result = result.numpy()
        return result

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        is_numpy = isinstance(x, np.ndarray)
        mat = self._convert_to_matrix(x)
        mat_inv = mat.transpose(-1, -2)
        result = self._convert_from_matrix(mat_inv)
        if is_numpy:
            result = result.numpy()
        return result


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-5

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
