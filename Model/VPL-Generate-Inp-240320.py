# -*- coding: utf-8 -*-
"""
@author: Liyuanxin
"""

'''
This code runs in pycharm 2023

path_input_inp is the input path
path_save_inp is the output path of the inp file
Ensure that the name of the input inp is VPL10u0000h20s0385n0190t-3m4
Then set the friction coefficient value
Scratch depth value
Yield stress value
value of hardening index
Subfolders will be generated according to friction coefficient 
Double-click the bat file under the subfolder to proceed
'''

import os
import numpy as np
import shutil

# Inp input path modified according to the file storage path

path_input_inp = r"E:\Temp\240320"

# The output path of the generated inp file

path_save_inp = r'E:\Temp\240320out'

# Friction coefficient range
pl_u = [0]
# Scratch depth range(mm)
pl_h = [0.02]
# yield stress range(MPa)
#stress = [100,385]
stress = np.arange(100,2000,200)
# Hardening index range
stress_n = [0.1,0.2,0.3,0.4,0.5]
# Elastic modulus,modified according to the material
E = 192000



# Get all files with the inp suffix in the specified directory
def GetFileName(file_dir):
    file_name = []

    for files in os.listdir(file_dir):
        if os.path.splitext(files)[1] == '.inp':

            file_name.append(os.path.splitext(files)[0])
            print(files)
    return file_name

# Read the first inp file

for root, dirs, files in os.walk(path_input_inp):
    for file in files:
        file_name = os.path.join(root, file)
        moduel_name = os.path.splitext(file)[0]
        #print(file_name)
k = 0
for u in pl_u:
    # generate folder

    second_path = 'u%.4d' % (int(u * 1000) )
    if os.path.exists(os.path.join(path_save_inp, second_path)):
        shutil.rmtree(os.path.join(path_save_inp, second_path))
    os.mkdir(os.path.join(path_save_inp, second_path))

    for h in pl_h:
        for s in stress:
            for n in stress_n:

                #inpfile = open(file_name,encoding='utf-8')
                inpfile = open(file_name)
                lines = inpfile.readlines()
                #print(lines)
                inpfile.close()
                # Change the coefficient of friction
                originstr1 = "*Friction\n"
                newstr1 = "*Friction\n %s,\n*Surface Behavior, pressure-overclosure=HARD\n"%u
                #print(lines)
                strindex1 = lines.index(originstr1)
                #print(strindex1)
                lines[strindex1 + 1:strindex1 + 3] = ''
                lines[strindex1] = newstr1
                # Change the indentation depth
                originstr2 = 'Set-10, 2, 2, -0.02\n'
                newstr2 = 'Set-10, 2, 2, -%s\n'%h
                strindex2 = lines.index(originstr2)
                #print(strindex2)
                lines[strindex2] = newstr2

                # Change yield stress and hardening index
                strain = s / E

                table_str = ''
                strain_table = np.concatenate([np.linspace(strain, 0.1, 10), np.linspace(0.11, 5, 10)], axis=0)
                for i in strain_table:
                    s1 = s * pow(i / strain, n)
                    table_str += '%s,%s\n'%(round(s1,3),round(i - strain,3))

                originstr3 = '*Plastic\n'
                strindex3 = lines.index(originstr3)
                #print(table_str)
                newstr3 = "*Plastic\n"+table_str
                lines[strindex3 + 1:strindex3 + 21] = ''
                lines[strindex3] = newstr3
                # Replace filename
                file_name_new1 = file_name.replace('u0000','u%.4d'%int(u*1000))
                file_name_new2 = file_name_new1.replace('h20','h%.2d'%int(h*1000))
                file_name_new3 = file_name_new2.replace('s0385', 's%.4d' % int(s))
                file_name_new4 = file_name_new3.replace('n0190', 'n%.4d' % int(n * 1000))

                (filepath, tempfilename) = os.path.split(file_name_new4)
                (filename, extension) = os.path.splitext(tempfilename)
                # Change job name
                originstr4 = '** Job name: %s Model name: %s\n' % (moduel_name, moduel_name)
                newstr4 = '** Job name: %s Model name: %s \n' % (filename, filename)

                strindex4 = lines.index(originstr4)
                # print(strindex2)
                lines[strindex4] = newstr4

                # save_name
                save_name = os.path.join(path_save_inp, second_path,tempfilename)
                print(save_name)
                newfile = open(save_name, "w")
                for newline in lines:
                    newfile.write(newline)
                newfile.close()
                print(u,h,s,n)
                k = k+1
                print(k)





