import os
import numpy as np
from pupyC3D import C3DFile

from c3dTools.force_plate import ForcePlate
from c3dTools.maths import matlab_resample, rotation_matrix, filter_data


def marker2trc(c3dfile: C3DFile, output, **kwargs):
    axis = kwargs.get('rotation_axis', '')
    angle = kwargs.get('rotation_angle', 0.)
    if axis != '':
        rot = rotation_matrix(axis, [angle])
    else:
        rot = np.identity(3)
    markers = kwargs.get('markers', c3dfile.get_point_names())
    if len(markers)>0:
        header = ["PathFileType\t4\t(X/Y/Z)\t%s\n" % os.path.basename(output),
                  "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
                  "%d\t%d\t%d\t%d\t%s\t%d\t%d\t%d\n" % (c3dfile.frame_rate, c3dfile.frame_rate, c3dfile.frame_count,
                                                        c3dfile.point_count, c3dfile.point_unit[0], c3dfile.frame_rate,
                                                        c3dfile.header['first_frame'], c3dfile.frame_count)]
        line = '\t\t\t'.join(markers)
        line = "Frame#\tTime\t" + line + '\n'
        header.append(line)
        line = ['\tX%d\tY%d\tZ%d' % (i + 1, i + 1, i + 1) for i in range(len(markers))]
        line = '\t' + ''.join(line) + '\n'
        header.append(line)

        data_type = ['d', '.4f'] + ['.4f'] * len(markers) * 3

        data2write = np.zeros((c3dfile.frame_count, 2 + 3 * len(markers)))
        data2write[:, 0] = np.arange(c3dfile.frame_count) + 1
        data2write[:, 1] = np.arange(c3dfile.frame_count) / c3dfile.frame_rate
        for i, name in enumerate(markers):
            data2write[:, (i + 1) * 3 - 1:(i + 1) * 3 + 2] = np.dot(rot, (
            c3dfile.get_point_data(name)[:, :3]).transpose()).transpose()
        write_file(output, header, data2write, data_type)



def force2mot(force_plates: list[ForcePlate], output, facq:float, fout:float, **kwargs):
    axis = kwargs.get('rotation_axis', '')
    angle = kwargs.get('rotation_angle', 0.)
    if axis != '':
        rot = rotation_matrix(axis, [angle])
    else:
        rot = np.identity(3)
    filter_analog = kwargs.get('filter', False)
    f_cutoff = kwargs.get('cutoff', 5.)

    frame_count = int(force_plates[0].raw_data.shape[0]/(facq/fout))
    header = ['name %s\n' % os.path.basename(output), 'datacolumns %d\n' % (9 * len(force_plates) + 1),
              'datarows %d\n' % frame_count, 'range 0 %f\n' % ((frame_count - 1) / fout),
              'endheader\n']
    line = ['time']
    for i in range(len(force_plates)):
        if i == 0:
            line += ['ground_force_vx', 'ground_force_vy', "ground_force_vz", 'ground_force_px', 'ground_force_py',
                     'ground_force_pz', 'ground_torque_x', 'ground_torque_y', 'ground_torque_z']
        else:
            line += ['%d_ground_force_vx' % (i + 1), '%d_ground_force_vy' % (i + 1), "%d_ground_force_vz" % (i + 1),
                     '%d_ground_force_px' % (i + 1), '%d_ground_force_py' % (i + 1), '%d_ground_force_pz' % (i + 1),
                     '%d_ground_torque_x' % (i + 1), '%d_ground_torque_y' % (i + 1), '%d_ground_torque_z' % (i + 1)]
    line = '\t'.join(line) + '\n'
    header.append(line)

    data_type = ['.8f'] * len(line)
    data2write = np.zeros((frame_count, 1 + 9 * len(force_plates)))
    data2write[:, 0] = np.arange(frame_count) / fout
    for i, pff in enumerate(force_plates):
        origin = np.mean(pff.corners, axis=0) / 1000 #assuming that positions are in millimeters
        force = matlab_resample(pff.get_force(), fout, facq)
        if filter_analog:
            force = filter_data(force, f_cutoff, fout)
        moment = matlab_resample(pff.get_moment() / 1000, fout, facq)
        if filter_analog:
            moment = filter_data(moment, f_cutoff, fout)
        moment = moment + np.array([np.cross(origin, force[i, :]) for i in range(force.shape[0])])
        cop = matlab_resample(pff.get_cop() / 1000, fout, facq)

        # force
        data2write[:, (i + 1) * 9 - 8:(i + 1) * 9 - 5] = np.dot(rot, force.transpose()).transpose()
        # cop
        data2write[:, (i + 1) * 9 - 5:(i + 1) * 9 - 2] = np.dot(rot, cop.transpose()).transpose()
        #  moment
        data2write[:, (i + 1) * 9 - 2:(i + 1) * 9 + 1] = np.dot(rot, moment.transpose()).transpose()
    data2write[np.where(np.isnan(data2write))] = 0
    write_file(output, header, data2write, data_type)


def write_file(filename, header, data, data_types):
    with open(filename, 'w') as f:
        for each in header:
            f.write(each)
        for i in range(data.shape[0]):
            line = ['%%%s' %s for s in data_types]
            line = [line[j] %each for j,each in enumerate(data[i,:])]
            f.write('\t'.join(line))
            f.write('\n')
