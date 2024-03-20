# -*- coding: utf-8 -*-
"""
@author: Liyuanxin
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *

import numpy as np

'''
The following code is running in abaqus 2020 
CPU is Intel i9-12900K
The unit of length is millimeters, and the unit of time is seconds.
'''


'''
Parameters that need to be modified , start
'''

'''

The scratch_length is scratch length in mm
The scratch_depth is scratch depth in mm , the negative sign indicates that the indenter is pressed down
The indentation_time represents the time for the indenter to press down in Step-1 , unit is s
The scratch_time represents the time for the indenter to move in Step-2 , unit is s

scratch_depth is the scratch depth. For the calculation example with a small grid, the scratch depth should be shallower, and the maximum scratch depth should not exceed 90um.
For the calculation example with a minimum mesh size of 5um, the scratch depth is not easy to exceed 50um.
For different indenters, the applicability may be different. In this case, it is a 136-degree Vickers indenter.
For different constitutions, the applicability may be different. This case is a power-law hardening constitutive

'''

scratch_length = 2
scratch_depth = -0.02
indentation_time = 1e-3
scratch_time = 1e-3

'''
x_width , y_heigth , z_length indicates the width, length and height of the matrix material 
It should be noted that z_length should be about 1mm larger than scratch_length
The encrypted area of the grid is generally 2 times the maximum grid size

Excessive area is generally 2 times the maximum mesh size, plus the maximum mesh size, so x_width generally take 5 times the maximum mesh size, 
can be greater than 5 times, not less than 5 times
y_heigth direction is generally 4 times the maximum mesh size, which needs to be greater than the scratch depth


'''

x_width = 1
y_heigth = 0.6
z_length = 2.9


'''
min_size is the minimum mesh size, the unit is mm, all mesh parameters are calculated according to the minimum mesh size, 
the minimum mesh size of 5um, the model generation takes about 30min

max_size the maximum grid size is equal to k * min_size, the standard value of the maximum grid size is 160um, generally greater than 100um, less than 200um
The value of k is raised to the nth power of 2, which is 2, 4, 8, 16, 32...

For example min_size = 0.015, k can take 8 min_size = 0.01, k can take 16 min_size = 0.005, k can take 32 to ensure 200um > k * min_size > 100um

'''

min_size = 0.01
k = 16

'''
Mass scaling is required for slow time
Mass1 is the indentation mass scaling factor, mass2 is the scratch mass scaling factor
Different mass scaling coefficients will affect the calculation results and need to be verified.

For different time orders, different mass scaling factors are required
Element-Element Method for Estimating Material Stability Time Steps
t = L/(E/p) ^ 0.5
t is the steady time step, E is the modulus of elasticity, p is the material density, and L is the length of the smallest unit
When the material density p is enlarged, the stabilization time step increases, reducing the calculation time
This is just a rough estimate


'''
mass1 = 1e4
mass2 = 1e4

'''
stress is Yield stress (MPa)
n is hardening index 
E is elastic modulus (MPa)
Nu is Poisson's ratio of materials
Mu is the coefficient of friction

numcpus = int (4) numcpus is the cpu for submitting the calculation. 
This is for the convenience of submitting the calculation. 
The number of cpu actually running in the end is based on the cpu of the bat file.

'''

stress = float(385)
n = 0.19
E = float(192000.0)
Nu = 0.3
Mu = 0.0

numcpus = int(4)

'''
Parameters that need to be modified , end
'''

# Calculate the maximum mesh size
max_size = k * min_size
# This parameter determines the mesh density in the z direction
maxsize =  1.0*max_size
# In order to ensure the consistency of the grid, here x_width, y_heigth, z_length are approximated to ensure that it is an integer multiple of the maximum size of the grid
k1 = int(round(x_width / max_size))
k2 = int(round(y_heigth / max_size))
k3 = int(round((z_length) / max_size))

if k3 % 2 == 0:
    k3 = k3
else:
    k3 = k3 + 1

# Get the width, height and length of the matrix and print

x1 = k1 * max_size
y1 = k2 * max_size
z1 = k3 * max_size
print(x1, y1, z1)

'''
# m = mdb.Model(name='VPL%.2du%.4dh%.2ds%.4dn%.4dt%sm%s')

Behind VPL is the grid size unit is um, behind u is the friction coefficient * 1000, behind h is the pressing depth unit is um,
s is followed by the yield stress unit is MPa, n is followed by the hardening index * 1000, 
t is time, m is the coefficient of mass scaling log (mass)

name is the model name, which is different for different mesh models, for example min_size = 0.01
The model name is: VPL10u0000h20s0385n0190t-3m4

Note: If you need this program to generate multiple inp files,do not modify any parameters.

'''
# model name
model_name = ('VPL%.2du%.4dh%.2ds%.4dn%.4dt%sm%s'
                  %(int(min_size*1000),int(Mu*1000),int(abs(scratch_depth*1000)),int(stress),int(n*1000),
                    int(np.log10(scratch_time)),int(np.log10(mass2))) )
# create the model
m = mdb.Model(name=model_name)



'''
# Create a 136-degree Vickers indenter start
# set geometry size

VickerAngle is Vickers indenter angle
Angle as face angle

z is the height of the indenter
x,y is the length and width 
'''
VickerAngle = 136


_angle = VickerAngle/2
z = 0.15
x = 2*np.tan(np.pi*_angle/180) * z
y = 2*np.tan(np.pi*_angle/180) * z

# creat Sketch
s1 = m.ConstrainedSketch(name='__profile__', sheetSize=200.0)
# rectangle
s1.rectangle(point1=(0.0, 0.0), point2=(x, y))

# creat  part-1 indenter deformed body

p1 = m.Part(name='Part-1', dimensionality=THREE_D, type=DEFORMABLE_BODY)

# BaseSolidExtrude
p1.BaseSolidExtrude(sketch=s1, depth=z)

# delet sketch
del m.sketches['__profile__']

# get geometry face index and edge index from coordinates point
f1, e1 = p1.faces, p1.edges
faces1 = f1.findAt((0, y / 2, z / 2))
edges1 = e1.findAt((0, y / 2, z))

# get the transform matrix
t1 = p1.MakeSketchTransform(sketchPlane=faces1, sketchUpEdge=edges1,
                            sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0, 0, 0))

s1 = m.ConstrainedSketch(name='__profile__',
                         sheetSize=0.4, gridSpacing=0.01, transform=t1)

# lines to cut
s1.Line(point1=(0, y), point2=(z, y))
s1.Line(point1=(z, y), point2=(z, y / 2))
s1.Line(point1=(z, y / 2), point2=(0, y))

# the first cut
p1.CutExtrude(sketchPlane=faces1, sketchUpEdge=edges1, sketchPlaneSide=SIDE1,
              sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)

del m.sketches['__profile__']

# the second get geometry face index and edge index from coordinates point get the transform matrix
f2, e2 = p1.faces, p1.edges
faces2 = f2.findAt((0, y / 2, z / 2))
edges2 = e2.findAt((0, y / 2, z))

t2 = p1.MakeSketchTransform(sketchPlane=faces2, sketchUpEdge=edges2,
                            sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0, 0, 0))
s2 = m.ConstrainedSketch(name='__profile__',
                         sheetSize=0.4, gridSpacing=0.01, transform=t2)
# lines to cut
s2.Line(point1=(0, 0), point2=(z, y / 2))
s2.Line(point1=(z, y / 2), point2=(z, 0))
s2.Line(point1=(z, 0), point2=(0, 0))
# the second cut
p1.CutExtrude(sketchPlane=faces2, sketchUpEdge=edges2, sketchPlaneSide=SIDE1,
              sketchOrientation=RIGHT, sketch=s2, flipExtrudeDirection=OFF)

del m.sketches['__profile__']

# the third get geometry face index and edge index from coordinates point get the transform matrix
f3, e3 = p1.faces, p1.edges

faces3 = f3.findAt((x / 2, y / 4, z / 2))
edges3 = e3.findAt((x, y / 4, z / 2))

t3 = p1.MakeSketchTransform(sketchPlane=faces3, sketchUpEdge=edges3, sketchPlaneSide=SIDE1, sketchOrientation=RIGHT,
                            origin=(0, 0, 0))

s3 = m.ConstrainedSketch(name='__profile__', sheetSize=0.4, gridSpacing=0.01, transform=t3)

y_s3 = pow(((y / 2) ** 2 + (z) ** 2), 0.5)

s3.Line(point1=(0, 0), point2=(0, y_s3))
s3.Line(point1=(0, y_s3), point2=(x / 2, y_s3))
s3.Line(point1=(x / 2, y_s3), point2=(0, 0))

# path edges
pathEdges = e3.findAt(((x, y / 2, 0),))

# the thrid cut
p1.CutSweep(path=pathEdges, sketchPlane=faces3, sketchUpEdge=edges3,
            sketchOrientation=RIGHT, profile=s3)

del m.sketches['__profile__']

# the fourth get geometry face index and edge index from coordinates point get the transform matrix
f4, e4 = p1.faces, p1.edges
faces4 = f4.findAt((x / 2, y / 4, z / 2))
edges4 = e4.findAt((x, y / 4, z / 2))

t4 = p1.MakeSketchTransform(sketchPlane=faces4, sketchUpEdge=edges4, sketchPlaneSide=SIDE1, sketchOrientation=RIGHT,
                            origin=(0, 0, 0))

s4 = m.ConstrainedSketch(name='__profile__', sheetSize=0.4, gridSpacing=0.01, transform=t4)

s4.Line(point1=(x, 0), point2=(x, y_s3))
s4.Line(point1=(x, y_s3), point2=(x / 2, y_s3))
s4.Line(point1=(x / 2, y_s3), point2=(x, 0))
# print faces4
# print edges4
# path edges
pathEdges1 = e4.findAt(((x, y / 2, 0),))

# the fourth cut
p1.CutSweep(path=pathEdges, sketchPlane=faces4, sketchUpEdge=edges4,
            sketchOrientation=RIGHT, profile=s4, flipSweepDirection=ON)

del m.sketches['__profile__']
# transform shell
c = p1.cells
# p1.RemoveCells(cellList = c[0:1])
p1.ReferencePoint(point=(x / 2, y / 2, z / 2))

'''
Create a 136-degree Vickers indenter end
'''

'''
Create matrix start
'''

import math

# creat Sketch
sp = m.ConstrainedSketch(name='__profile__', sheetSize=200.0)

# rectangle
sp.rectangle(point1=(0.0, 0.0), point2=(x1, y1))

# creat part-2
p2 = m.Part(name='Part-2', dimensionality=THREE_D, type=DEFORMABLE_BODY)

# BaseSolidExtrude
p2.BaseSolidExtrude(sketch=sp, depth=z1)

p2.DatumPointByCoordinate(coords=(0.0, 0.0, 0.0))

# delet sketch
del m.sketches['__profile__']

# Select surface to extract coordinates

f2 = p2.faces
faces = f2.findAt(((x1 / 2, y1, z1 / 2),), ((x1 / 2, y1, z1 / 2),))
p2.Set(faces=faces, name='Set-1')

'''
Create matrix end
'''
# partition the matrix start

# creat datum plane to partition the part2 first partition start
x_value = 2 * max_size


offset_value = y1 - 2 * max_size


# second partition start

# get geometry face index and edge index from coordinates point
fp2, ep2, dp2 = p2.faces, p2.edges, p2.datums
facesp2 = fp2.findAt((x1 / 2, y1, z1 / 2))
edgesp2 = ep2.findAt((x1 / 2, y1, z1))

# get the transform matrix
tp2 = p2.MakeSketchTransform(sketchPlane=facesp2, sketchUpEdge=edgesp2,
                             sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0, y1, 0))

sp2 = m.ConstrainedSketch(name='__profile__',
                          sheetSize=40 * max_size, gridSpacing=max_size, transform=tp2)

gp2, vp2, dp2, cp2 = sp2.geometry, sp2.vertices, sp2.dimensions, sp2.constraints

#  partition
def Refined_Mesh(min_mesh_size, max_mesh_size, index):

    global oy
    minms = min_mesh_size
    maxms = max_mesh_size
    index = int(index)
    baisvalue = 2 ** index * minms
    # oringin x and y
    ox = 0
    oy = x_value + baisvalue

    sp2.rectangle(point1=(ox, oy), point2=(ox + baisvalue, oy + baisvalue))
    sp2.Line(point1=(ox, oy + baisvalue), point2=(ox + baisvalue / 2, oy + baisvalue / 2))
    sp2.Line(point1=(ox + baisvalue / 2, oy), point2=(ox + baisvalue / 2, oy + baisvalue / 2))
    sp2.Line(point1=(ox + baisvalue, oy + baisvalue / 2), point2=(ox + baisvalue / 2, oy + baisvalue / 2))

    lp2 = gp2.findAt((ox + baisvalue, oy + baisvalue / 4))
    print(lp2.id)
    # print gp2
    id = int(lp2.id)
    gp2_mirror = (gp2[id - 2], gp2[id - 1], gp2[id + 1], gp2[id + 2], gp2[id + 3], gp2[id + 4])

    # mirror
    sp2.copyMirror(mirrorLine=lp2, objectList=gp2_mirror)

    sp2.linearPattern(geomList=(gp2[id - 2], gp2[id - 1], gp2[id + 0], gp2[id + 1], gp2[id + 2], gp2[id + 3],
                                gp2[id + 4], gp2[id + 5], gp2[id + 6], gp2[id + 7], gp2[id + 8], gp2[id + 9],
                                gp2[id + 10]),
                      vertexList=(), number1=int(z1 / (4 * minms)), spacing1=2 ** index * minms * 2, angle1=0.0,
                      number2=1, spacing2=0.665, angle2=90.0)

for i in range(0, int(math.log(int(max_size / min_size), 2))):

    print(i)

    Refined_Mesh(min_size, max_size, i + 1)

# chose the face for partition
pickedFaces = fp2.findAt(((x1 / 2, y1, z1 / 2),))
p2.PartitionFaceBySketch(sketchUpEdge=edgesp2, faces=pickedFaces, sketch=sp2)

del m.sketches['__profile__']

# second partition end

# partition end

# assembly start
# assembly part1 and part2

a = m.rootAssembly

a.DatumCsysByDefault(CARTESIAN)

a.Instance(name='Part-1-1', part=p1, dependent=OFF)
a.Instance(name='Part-2-1', part=p2, dependent=OFF)

a.rotate(instanceList=('Part-1-1',), axisPoint=(0.0, y / 2, 0.0),
         axisDirection=(x, 0.0, 0.0), angle=90.0)
a.translate(instanceList=('Part-1-1',), vector=(-x / 2, y1 - (y / 2 - z), 0.0))

a.rotate(instanceList=('Part-1-1',), axisPoint=(0.0, y, 0.0),
         axisDirection=(0.0, y, 0.0), angle=45.0)

start_location = (z1 - scratch_length) * 0.5 + min_size / 2

a.translate(instanceList=('Part-1-1',), vector=(0, 0, start_location))

# partition
k3 = 6
a = m.rootAssembly
c1 = a.instances['Part-2-1'].cells
pickedCells = c1.findAt(((0, 0, 0),),)
e1 = a.instances['Part-2-1'].edges
v1 = a.instances['Part-2-1'].vertices
e1 = a.instances['Part-2-1'].edges


edges = e1.findAt(((oy, y1,z1-k3*max_size+min_size/2), ), )
point = v1.findAt(((oy, y1,z1-k3*max_size), ), )

a.PartitionCellByPlanePointNormal(point=point[0], normal=edges[0],cells=pickedCells)

# The creation surface is used to extract the node stress-strain

f1 = a.instances['Part-2-1'].faces
faces1 = f1.findAt(((oy+min_size/2, y1/2,z1-k3*max_size), ), )
a.Set(faces=faces1, name='Set-5')
# assembly end

# mesh start

# grid the indetation part-1
partInstances_1 = (a.instances['Part-1-1'],)

cp1 = a.instances['Part-1-1'].cells
cellsp1 = cp1.findAt(((0, y1 + z / 2, start_location),))

a.setMeshControls(regions=cellsp1, elemShape=TET, technique=FREE)

a.seedPartInstance(regions=partInstances_1, size=0.02, deviationFactor=0.1, minSizeFactor=0.1)

elemType1 = mesh.ElemType(elemCode=UNKNOWN_HEX, elemLibrary=EXPLICIT)
elemType2 = mesh.ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT)
elemType3 = mesh.ElemType(elemCode=C3D10M, elemLibrary=EXPLICIT,
                          secondOrderAccuracy=OFF, distortionControl=DEFAULT)

pickedRegions = (cellsp1,)
a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, elemType3))

a.generateMesh(regions=partInstances_1)

# grid the matrix  part-2
# choose the edges to seed min-size mesh
mesh_e = a.instances['Part-2-1'].edges
pickedEdges_min = mesh_e.findAt(((0, y1, z1 / 2),), ((x_value / 2, y1, 0),),((0, y1, z1-k3*max_size/2),))
a.seedEdgeBySize(edges=pickedEdges_min, size=min_size, deviationFactor=0.1, constraint=FINER)
# choose the edges to seed max-size mesh

pickedEdges_max = mesh_e.findAt(((x1, y1, z1 / 2),), ((x1 - x_value / 2, y1, 0),))

a.seedEdgeBySize(edges=pickedEdges_max, size=max_size, deviationFactor=0.1, constraint=FINER)


# the layer number of the grid

pickedEdges_num = mesh_e.findAt(((0, y1 / 2, 0),))
# number=20
# a.seedEdgeByBias(biasMethod=SINGLE, end2Edges=pickedEdges_num, ratio=2*y/(number*min_size), number=20, constraint=FINER)

a.seedEdgeByBias(biasMethod=SINGLE, end2Edges=pickedEdges_num, minSize=min_size, maxSize=maxsize/2, constraint=FINER)

partInstances2 = (a.instances['Part-2-1'],)
a.seedPartInstance(regions=partInstances2, size=max_size, deviationFactor=0.1,
                   minSizeFactor=0.1)

# elem type

elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=EXPLICIT,
    kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF,
    hourglassControl=ENHANCED, distortionControl=ON, lengthRatio=0.5)

elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=EXPLICIT)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=EXPLICIT)

c1 = a.instances['Part-2-1'].cells

pickedRegions = (c1.findAt(((0, 0, 0),), ((0, y1, z1),)), )
a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, elemType3))

# grid mesh
# a.generateMesh(regions=pickedRegions)

partInstances_all = (a.instances['Part-2-1'],)
a.generateMesh(regions=partInstances_all, meshTechniqueOverride=ON)
# mesh end


# material start
m.Material(name='Material-1')

import numpy as np

strain = stress / E
table = []
strain_table = np.concatenate([np.linspace(strain, 0.1, 10), np.linspace(0.11, 5, 10)], axis=0)
for i in strain_table:
    s = stress * pow(i / strain, n)
    table.append((round(s,3), round(i - strain,3) ))

table = tuple(table)


m.materials['Material-1'].Density(table=((7.8e-09,),))
m.materials['Material-1'].Elastic(table=((E, Nu),))
m.materials['Material-1'].Plastic(table=table)

m.HomogeneousSolidSection(name='Section-1',material='Material-1', thickness=None)

# assign the material to indenter


cellsp1 = c.findAt(((0, 0, 0),))

region = p1.Set(cells=cellsp1, name='Set-1')

p1.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                     offsetType=MIDDLE_SURFACE, offsetField='',
                     thicknessAssignment=FROM_SECTION)

cp2 = p2.cells

cellsp2 = cp2.findAt(((0, 0, 0),))
print
cellsp2
# assign the material to matrix
region = p2.Set(cells=cellsp2, name='Set-2')

p2.SectionAssignment(region=region, sectionName='Section-1', offset=0.0,
                     offsetType=MIDDLE_SURFACE, offsetField='',
                     thicknessAssignment=FROM_SECTION)

# material end

# step start

m.ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=indentation_time,
                       massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                                     mass1, 0.0, None, 0, 0, 0.0, 0.0, 0, None),))

m.ExplicitDynamicsStep(name='Step-2', previous='Step-1', timePeriod=scratch_time,
                       massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
                                     mass2, 0.0, None, 0, 0, 0.0, 0.0, 0, None),))

m.fieldOutputRequests['F-Output-1'].setValuesInStep(stepName='Step-2', numIntervals=100)
m.fieldOutputRequests['F-Output-1'].setValues(
    variables=('S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U',
               'V', 'A', 'RF', 'CSTRESS', 'EVF', 'COORD'))

# step end
print('step end')

# interaction start
# creat the contact propery
m.ContactProperty('IntProp-1')
m.interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((Mu,),), shearStressLimit=None,
    maximumElasticSlip=FRACTION,
    fraction=0.005, elasticSlipStiffness=None)
m.interactionProperties['IntProp-1'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON,
    constraintEnforcementMethod=DEFAULT)

s1 = a.instances['Part-1-1'].faces
s2 = a.instances['Part-2-1'].faces

# creat the surface to surface contact

point1 = (0, y1, start_location)
point2 = (0, y1 + z, start_location - pow((x ** 2 + y ** 2), 0.5) / 2)
point3 = (- pow((x ** 2 + y ** 2), 0.5) / 2, y1 + z, start_location)
point4 = (pow((x ** 2 + y ** 2), 0.5) / 2, y1 + z, start_location)
point5 = (0, y1 + z, start_location + pow((x ** 2 + y ** 2), 0.5) / 2)

face_point1 = ((point1[0] + point2[0] + point3[0]) / 3, (point1[1] + point2[1] + point3[1]) / 3,
               (point1[2] + point2[2] + point3[2]) / 3)
face_point2 = ((point1[0] + point2[0] + point4[0]) / 3, (point1[1] + point2[1] + point4[1]) / 3,
               (point1[2] + point2[2] + point4[2]) / 3)
face_point3 = ((point1[0] + point3[0] + point5[0]) / 3, (point1[1] + point3[1] + point5[1]) / 3,
               (point1[2] + point3[2] + point5[2]) / 3)
face_point4 = ((point1[0] + point4[0] + point5[0]) / 3, (point1[1] + point4[1] + point5[1]) / 3,
               (point1[2] + point4[2] + point5[2]) / 3)

side1Faces1 = s1.findAt((face_point1,), (face_point2,), (face_point3,), (face_point4,))

region1 = a.Surface(side1Faces=side1Faces1, name='m_Surf-1')

side2Faces1 = s2.findAt(((x_value / 2, y1, z1 / 2),),((x_value / 2, y1, z1-max_size),))
region2 = a.Surface(side2Faces=side2Faces1, name='s_Surf-1')

print
side1Faces1

m.SurfaceToSurfaceContactExp(name='Int-1',
                             createStepName='Initial', master=region1, slave=region2,
                             mechanicalConstraint=KINEMATIC, sliding=FINITE,
                             interactionProperty='IntProp-1', initialClearance=OMIT, datumAxis=None,
                             clearanceRegion=None)

# get the contact area
m.HistoryOutputRequest(
    name='H-Output-2', createStepName='Step-1', variables=('CFNM', 'CFN1',
                                                           'CFN2', 'CFN3', 'CFSM', 'CFS1', 'CFS2', 'CFS3', 'CFTM',
                                                           'CFT1', 'CFT2',
                                                           'CFT3', 'CAREA'), interactions=('Int-1',),
    sectionPoints=DEFAULT,
    rebar=EXCLUDE)

# interaction end

# define mass and heat of reference point
r1 = a.instances['Part-1-1'].referencePoints
print
r1.keys()
number = int(r1.keys()[0])

refPoints1 = (r1[number],)

region = a.Set(referencePoints=refPoints1, name='Set-7')

region2 = a.instances['Part-1-1'].sets['Set-1']

region1 = a.sets['Set-7']
m.RigidBody(name='Constraint-1', refPointRegion=region1, bodyRegion=region2)

# interaction end
print('interaction end')

# load start
# Xsymm
f1 = a.instances['Part-2-1'].faces

faces1 = f1.findAt(((0, (y1 - offset_value) / 2, z1 / 2),), ((0, y1 - offset_value / 2, z1 / 2),) , ((0, y1 - offset_value / 2, z1-k3*max_size/2),) )
region_x = a.Set(faces=faces1, name='Set-8')

m.XsymmBC(name='BC-1', createStepName='Initial', region=region_x, localCsys=None)

#  Fixed BC
matrixface1 = (x1 / 2, (y1 - offset_value) / 2, 0)

matrixface2 = (x1 / 2, y1 - offset_value / 2, 0)

matrixface3 = (x1 / 2, (y1 - offset_value) / 2, z1)

matrixface4 = (x1 / 2, y1 - offset_value / 2, z1)

matrixface5 = (x1, (y1 - offset_value) / 2, z1 / 2)

matrixface6 = (x1, y1 - offset_value / 2, z1 / 2)

matrixface7 = (x1 / 2, 0, z1 / 2)

matrixface8 = (x1 , y1 - offset_value / 2, z1-k3*max_size/2)

matrixface9 = (x1/2 , 0, z1-k3*max_size/2)


faces1 = f1.findAt((matrixface1,), (matrixface3,), (matrixface5,), (matrixface7,) , (matrixface8,),(matrixface9,))


region_f = a.Set(faces=faces1, name='Set-9')
m.EncastreBC(name='BC-2', createStepName='Initial', region=region_f, localCsys=None)

# refercepoint BC
region_r = a.Set(referencePoints=refPoints1, name='Set-10')
m.DisplacementBC(name='BC-3',
                 createStepName='Initial', region=region_r, u1=SET, u2=UNSET, u3=UNSET,
                 ur1=SET, ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM,
                 fieldName='', localCsys=None)
# indent
# scratch

m.TabularAmplitude(name='Amp-1', timeSpan=TOTAL,
                   smooth=SOLVER_DEFAULT,
                   data=((0.0, 0.0), (indentation_time, 1.0), (indentation_time + scratch_time, 1.0)))

m.TabularAmplitude(name='Amp-2', timeSpan=STEP,
                   smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (scratch_time, 1.0)))

m.DisplacementBC(name='BC-4',
                 createStepName='Step-1', region=region_r, u1=UNSET, u2=scratch_depth, u3=UNSET,
                 ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude='Amp-1', fixed=OFF,
                 distributionType=UNIFORM, fieldName='', localCsys=None)

m.DisplacementBC(name='BC-5',
                 createStepName='Step-2', region=region_r, u1=UNSET, u2=UNSET, u3=scratch_length,
                 ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude='Amp-2', fixed=OFF,
                 distributionType=UNIFORM, fieldName='', localCsys=None)
# load end

print('Model created successfully')

mdb.Job(name=model_name, model=model_name, description='', type=ANALYSIS,
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, explicitPrecision=DOUBLE_PLUS_PACK, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF,
    historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, 
    parallelizationMethodExplicit=DOMAIN, numDomains=numcpus,
    activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=numcpus)


'''
mdb.jobs[model_name].submit(consistencyChecking=OFF)

'''

