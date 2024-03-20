# -*- coding: utf-8 -*-
"""
@author: Liyuanxin
"""
'''
This code runs in abaqus 2020
'''
import os, os.path, sys
from abaqus import *
from odbAccess import *
from abaqusConstants import *
from caeModules import *
import numpy as np


# Define a function, output coordinates, normal force, contact area, ALLAE/ALLIE

def field_out(file_name):
    myodb = session.openOdb(file_name)

    myViewports = session.viewports['Viewport: 1']
    myViewports.setValues(displayedObject=myodb)
    step2 = myodb.steps['Step-2']
    lastFrame = step2.frames[-1]
    region = myodb.rootAssembly.instances['PART-2-1'].nodeSets['SET-1']
    coords = lastFrame.fieldOutputs['COORD'].getSubset(region=region)
    coordvalue = coords.values

    # Get the node number of set-5
    region1 = myodb.rootAssembly.nodeSets['SET-5']
    coords_section = lastFrame.fieldOutputs['COORD'].getSubset(region=region1)
    coordvalue_section = coords_section.values
    coords_section_data = []
    for j in coordvalue_section:
        coords_section_data.append([j.nodeLabel, j.data[0], j.data[1], j.data[2]])
    # print(coordvalue)

    coordata = []
    for i in coordvalue:
        # print(i)
        coordata.append([i.nodeLabel, i.data[0], i.data[1], i.data[2]])

    force = session.xyDataListFromField(odb=myodb, outputPosition=NODAL,
                                        variable=(
                                            ('RF', NODAL,
                                             ((COMPONENT, 'RF1'), (COMPONENT, 'RF2'), (COMPONENT, 'RF3'),)),),
                                        nodeSets=("SET-10",))

    base_name = os.path.splitext(file_name)[0]
    print(base_name)
    # Select set5
    leaf = dgo.LeafFromNodeSets(nodeSets=("SET-5",))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
    # Set the format of the output
    nf = NumberFormat(numDigits=9, precision=0, format=AUTOMATIC)
    session.fieldReportOptions.setValues(numberFormat=nf)
    session.fieldReportOptions.setValues(printTotal=OFF, printMinMax=OFF,
                                         reportFormat=COMMA_SEPARATED_VALUES)
    # read
    session.writeFieldReport(fileName='%s.csv' % base_name, append=OFF,
                             sortItem='Node Label', odb=myodb, step=1, frame=100, outputPosition=NODAL,
                             variable=(('COORD', NODAL, ((COMPONENT, 'COOR1'), (COMPONENT, 'COOR2'), (
                                 COMPONENT, 'COOR3'),)), ('PE', INTEGRATION_POINT, ((COMPONENT, 'PE11'), (
                                 COMPONENT, 'PE22'), (COMPONENT, 'PE12'),)), ('PEEQ', INTEGRATION_POINT),
                                       ), stepFrame=SPECIFY)

    # np.savetxt('%s-set5.csv'%base_name,coords_section_data,fmt='%.8f', delimiter=',')

    try:

        area_data = session.XYDataFromHistory(
            name='CAREA    ASSEMBLY_M_SURF-1/ASSEMBLY_S_SURF-1', odb=myodb,
            outputVariableName='Total area in contact: CAREA    ASSEMBLY_M_SURF-1/ASSEMBLY_S_SURF-1',
            steps=('Step-1', 'Step-2',), __linkedVpName__='Viewport: 1')

        ALLAE = session.XYDataFromHistory(odb=myodb, name='ALLAE for Whole Model',
                                          outputVariableName='Artificial strain energy: ALLAE for Whole Model',
                                          steps=('Step-1', 'Step-2',), suppressQuery=True,
                                          __linkedVpName__='Viewport: 1')

        ALLIE = session.XYDataFromHistory(odb=myodb, name='ALLIE for Whole Model',
                                          outputVariableName='Internal energy: ALLIE for Whole Model',
                                          steps=('Step-1', 'Step-2',), suppressQuery=True)

    except:
        print('dont have the contact area')
    myodb.close()
    return coordata, force, area_data, ALLAE / ALLIE


import numpy as np

'''
Read the coordinate data
#min_size = 0.01
#max_size = 16 * min_size
#scratch_length = 2
#18 * max_size

Consistent with the parameters in ScratchVPL-240320
The extracted scratch depth and scratch width is the average of 0.6-0.8 of the scratch length

'''

def read_data(coor_data):
    surface = np.array(coor_data)
    # choose the suitable data  0.6-0.8 scratch length
    min_size = 0.01
    max_size = 16 * min_size
    scratch_length = 2
    start_position = (18 * max_size - scratch_length) / 2 + min_size / 2
    start_location = start_position + scratch_length * 0.6
    end_location = start_position + scratch_length * 0.8
    start_arg = np.where((surface[:, 3] >= start_location) & (surface[:, 3] <= end_location))[0]
    # data
    ymaxvalue = np.max(surface[start_arg, 2])
    yminvalue = np.min(surface[start_arg, 2])

    print(ymaxvalue - 0.1 * min_size, yminvalue + 0.1 * min_size, start_location, end_location)

    y_max_arg = \
        np.where((surface[start_arg, 2] >= ymaxvalue - 0.05 * min_size) & (surface[start_arg, 2] <= ymaxvalue))[0]
    y_min_arg = np.where((surface[start_arg, 2] >= yminvalue) & (surface[start_arg, 2] < yminvalue + 0.05 * min_size))[
        0]

    scratch_depth = np.mean(surface[start_arg[y_max_arg], 2]) - np.mean(surface[start_arg[y_min_arg], 2])

    scratch_width = np.mean(surface[start_arg[y_max_arg], 1]) - np.mean(surface[start_arg[y_min_arg], 1])

    scratch_hp = np.mean(surface[start_arg[y_max_arg], 2]) - 0.64

    print(scratch_depth * 1000, scratch_width * 1000)

    return [scratch_depth * 1000, scratch_width * 1000, scratch_hp * 1000]


# The directory where odb is stored is also the output directory of the data
# Modify the path where the file is located to read all odb files in the folder

path = r"E:\Temp\240320out\u0000"

header = ('u', 'h', 'stress', 'n', 'hrp', 'hp', 'w', 'fn', 'ft', 'area', 'energy_ratio')
import csv


f = open("%s/all_data.csv" % path, "wb")

writer = csv.writer(f)
writer.writerow(header)

log_file = []
for root, dirs, files in os.walk(path):
    for file in files:
        if os.path.splitext(file)[1] == '.odb':
            file_name = os.path.join(root, file)
            num = file.strip('.odb')
            # print(path)
            coordvalue, force, areadata, energyratio = field_out(file_name)
            surface = read_data(coordvalue)
            # print([surface[0],surface[1],strainmax])
            force3 = np.concatenate((np.array(force[0]), np.array(force[1]), np.array(force[2])), axis=1)
            area = np.array(areadata)
            energy = np.array(energyratio)
            # print(force3[80:90,[3,5]])
            forcemean = np.mean(force3[80:110, [3, 5]], axis=0)
            areamean = np.mean(area[int(0.8 * len(areadata)):int(0.9 * len(areadata)), 1])
            energymean = np.mean(energy[int(0.8 * len(energyratio)):int(0.9 * len(energyratio)), 1])

            print(forcemean, areamean)
            np.savetxt('%s\%s-coor.csv' % (path, num), np.array(coordvalue), fmt='%.8f', delimiter=',')
            np.savetxt('%s\%s-force.csv' % (path, num), np.array(force3), fmt='%.8f', delimiter=',')
            np.savetxt('%s\%s-area.csv' % (path, num), np.array(areadata), fmt='%.8f', delimiter=',')
            np.savetxt('%s\%s-surface.csv' % (path, num), np.array([surface[0], surface[1]]).reshape(1, 2), fmt='%.8f',
                       delimiter=',')
            np.savetxt('%s\%s-energy.csv' % (path, num), np.array(energyratio), fmt='%.8f', delimiter=',')
            log_file.append("%s is read successfully" % num)
            arg_s = num.rfind('s')
            s = float(num[arg_s + 1:arg_s + 5])
            arg_n = num.rfind('n')
            n = float(num[arg_n + 1:arg_n + 5]) / 1000
            arg_u = num.rfind('u')
            u = float(num[arg_u + 1:arg_u + 5]) / 1000
            arg_h = num.rfind('h')
            h = float(num[arg_h + 1:arg_h + 3])
            writer.writerow(
                [u, h, s, n, surface[0], surface[2], surface[1], -forcemean[0], forcemean[1], areamean, energymean])
            print("%s is read successfully" % num)

f.close()
np.savetxt('%s\log_file.txt' % path, np.array(log_file, dtype='object'), fmt="%s")















