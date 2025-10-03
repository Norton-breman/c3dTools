import warnings
import numpy as np
from pupyC3D import C3DFile


def extract_force_plates(c3d_file: C3DFile):
    group = c3d_file.get_parameter_group('FORCE_PLATFORM')
    if group is not None:
        fps = []
        nb_pff = c3d_file.get_parameter('FORCE_PLATFORM', 'USED').value
        all_corners = c3d_file.get_parameter('FORCE_PLATFORM', 'CORNERS').value
        all_origin = c3d_file.get_parameter('FORCE_PLATFORM', 'ORIGIN').value
        all_channels = c3d_file.get_parameter('FORCE_PLATFORM', 'CHANNEL').value
        all_calib = c3d_file.get_parameter('FORCE_PLATFORM', 'CAL_MATRIX').value
        for i in range(nb_pff):
            if nb_pff == 1:
                pff_type = c3d_file.get_parameter('FORCE_PLATFORM', 'TYPE').value
            else:
                pff_type = c3d_file.get_parameter('FORCE_PLATFORM', 'TYPE').value[i]
            corners = all_corners[i, :, :]

            origin = all_origin[i,:]
            if len(all_calib) > 0:
                calib = all_calib[i, :, :].transpose()
            else:
                calib = np.zeros((6, 6))
            channels = all_channels[i,:] - 1
            pff_desc = c3d_file.get_parameter('ANALOG', 'DESCRIPTIONS')
            if pff_desc is not None:
                pff_name = pff_desc.value[channels[0]]
            else:
                pff_name = ''
            labels = [c3d_file.get_analog_names()[j] for j in channels]
            data = np.zeros((c3d_file.analog_frame_count, len(labels)))
            for k, name in enumerate(labels):
                data[:, k] = c3d_file.get_analog_data(name).reshape((c3d_file.analog_frame_count,))
            fps.append(ForcePlate(pff_type, nb_frames=c3d_file.analog_frame_count, corners=corners, origin=origin,
                                  cal_matrix=calib, labels=labels, data=data, description=pff_name))
    else:
        warnings.warn('Can not find FORCE_PLATFORM group in the file %s' % c3d_file.filename)
        fps = []

    return fps

class ForcePlate:

    def __init__(self, pff_type, **kwargs):
        if pff_type not in [1, 2, 3, 4]:
            ValueError('PFF type not supported')
        self.type = pff_type
        self.description = kwargs.get('description','')
        self.nb_frames = kwargs.get('nb_frames',0)
        self.origin = kwargs.get('origin', np.zeros((1,3)))
        self.corners = kwargs.get('corners', np.zeros((4, 3)))
        self.raw_data = kwargs.get('data', np.zeros((self.nb_frames, 0)))
        self.labels = kwargs.get('labels', [])
        self.cal_matrix = kwargs.get('cal_matrix', np.zeros((6, 6)))
        if self.type == 4:
            self.raw_data = np.dot(self.cal_matrix, self.raw_data.transpose()).transpose()

    def get_raw_force(self):
        if self.type == 3:
            raw_force = np.zeros((self.nb_frames, 3))
            raw_force[:, 0] = np.sum(self.raw_data[:, :2], axis=1)
            raw_force[:, 1] = np.sum(self.raw_data[:, 2:4], axis=1)
            raw_force[:, 2] = np.sum(self.raw_data[:, 4:8], axis=1)
            return raw_force
        else:
            return np.copy(self.raw_data[:, :3])

    def get_raw_cop(self):
        if not self.type == 1:
            raise ValueError('PFF with type %d do not contain raw cop' % self.type)
        cop_raw = np.zeros((self.nb_frames, 3))
        cop_raw[:, :2] = np.copy(self.raw_data[:, 3:5])
        return cop_raw

    def get_raw_tz(self):
        if not self.type == 1:
            raise ValueError('PFF with type %d do not contain raw tz' % self.type)
        return np.copy(self.raw_data[:, 5])

    def get_raw_moment(self):
        if self.type == 1:
            raise ValueError('PFF with type %d do not contain raw moment' % self.type)
        else:
            raw_moment = np.zeros((self.nb_frames, 3))
            if self.type == 3:
                raw_moment[:, 0] = self.origin[0, 1] * (
                            np.sum(self.raw_data[:, 4:6], axis=1) - np.sum(self.raw_data[:, 6:8], axis=1))
                raw_moment[:, 1] = self.origin[0, 0] * (
                        np.sum(self.raw_data[:, 5:7], axis=1) - self.raw_data[:, 4] - self.raw_data[:, 7])
                raw_moment[:, 0] = self.origin[0, 1] * (self.raw_data[:, 1] - self.raw_data[:, 0]) - self.origin[
                    0, 0] * (self.raw_data[:, 2] - self.raw_data[:, 3])
                raw_moment += np.cross(self.get_raw_force(), np.array([0, 0, self.origin[0, 2]]))
            else:
                raw_moment = np.copy(self.raw_data[:, 3:6])
                raw_moment += np.cross(self.get_raw_force(), self.origin)
            return raw_moment

    def get_force(self):
        return np.dot(self.get_reference_frame().transpose(), self.get_raw_force().transpose()).transpose()

    def get_moment(self):
        if self.type == 1:
            cop = np.dot(self.get_reference_frame().transpose(), self.get_raw_cop().transpose()).transpose()
            return np.cross(self.get_force(), cop - self.get_tz())
        else:
            return np.dot(self.get_reference_frame().transpose(), self.get_raw_moment().transpose()).transpose()

    def get_cop(self):
        if self.type == 1:
            return np.dot(self.get_reference_frame().transpose(), self.get_raw_cop().transpose()).transpose() + np.mean(
                self.corners, 0)
        else:
            moments = self.get_raw_moment()
            forces = self.get_raw_force()

            cop = np.cross(forces, moments) / np.pow(np.linalg.norm(forces, axis=1), 2)[:, np.newaxis]
            cop = cop - ((cop[:, 2] / forces[:, 2]) * forces.transpose()).transpose()
            return np.dot(self.get_reference_frame().transpose(), cop.transpose()).transpose() + np.mean(self.corners,
                                                                                                         0)

    def get_tz(self):
        if self.type == 1:
            tz = np.zeros((self.nb_frames, 3))
            tz[:, 2] = self.get_raw_tz()
            return np.dot(self.get_reference_frame().transpose(), tz.transpose()).transpose()
        else:
            moments = self.get_raw_moment()
            forces = self.get_raw_force()
            cop = np.zeros((self.nb_frames, 3))
            cop[:, 0] = -moments[:, 1] / forces[:, 2]
            cop[:, 1] = moments[:, 0] / forces[:, 2]
            return np.dot(self.get_reference_frame().transpose(),
                          (moments - np.cross(forces, -cop)).transpose()).transpose()

    def get_reference_frame(self):
        x = self.corners[0, :] - self.corners[1, :]
        y = self.corners[0, :] - self.corners[3, :]
        z = np.cross(x, y)
        y = np.cross(z, x)
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)
        ref_frame = np.zeros((3, 3))
        ref_frame[0, :] = x
        ref_frame[1, :] = y
        ref_frame[2, :] = z
        return ref_frame