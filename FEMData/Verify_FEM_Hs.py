from builtins import print
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
import scipy.stats as st


def ReadCsv(filename):
    data = pd.read_csv(filename)
    #print(data.columns)
    return data

data = ReadCsv('FEM50.csv').iloc[0:50]
print(data)

def calculation_parameter(data,kn1,kn,kt):
    f_pre = []
    for h, w, s, n, hrp, hp, ft, u, fn, E in zip(data['h']*1e-6, data['w']*1e-6, data['stress']*1e6, data['n'], data['hrp']*1e-6, data['hp']*1e-6,data['ft']*2, data['u'], data['fn']*2, data['E']*1e9):

        hr = hrp - hp
        stress_y = s
        strain_y = (stress_y / E)
        #print(strain_y)
        v = 8.1681 * pow(hrp, 3)
        vp = 2/3*(w-3.5*(hr))*4.95*hr*(hp)
        #print(v,vp)
        theta = 148.11/180*np.pi
        c_138 = ((pow( (v-vp) / (np.pi * strain_y), 1 / 3))) / (1 - np.cos(theta/2))
        #print(c_138)
        a = hrp
        #print(c_138,w,hrp)
        arg_fn = np.pi * E * strain_y * pow(np.sin(theta/2) * c_138, 3 * n)/(2-3*n)
        #print(arg_fn)
        rn1 =  hrp
        rn2 = kn * w
        fn_pre = arg_fn * (pow(rn2, 2 - 3 * n) - pow(rn1, 2 - 3 * n)) * pow(kn1, 2 - 3 * n)/np.sin(theta / 2)


        H_s = (3.32-5.79*n+2.8*pow(n,2))*pow(0.91*s/E,0.07-1.283*n+0.248*n*n)

        f_pre.append(np.array([float(0.91*fn_pre/(w*w*s))  , H_s  ]).T )

    return np.array(f_pre)

kn1,kn = 1.0,1.52
kt = 0.59

f = calculation_parameter(data,kn1,kn,kt)

x_n = np.array(f[:,0]).flatten()
y_n = np.array(f[:,1]).flatten()

hs_error_fem = 100 * abs( f[:, 0]  - f[:, 1]) / f[:,1]

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,4))

subplot_labels = ['(a)', '(b)']

plt.subplot(121)

plt.text(-0.15, 1, subplot_labels[0], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

n01 = np.arange(0,50,5)
n02 = np.arange(1,50,5)
n03 = np.arange(2,50,5)
n04 = np.arange(3,50,5)
n05 = np.arange(4,50,5)

color = ['r', 'g', 'b', 'c', 'k','m']
marker = ['o','v','<','>','s','+']
type1 = ['ro-', 'gv--', 'b<:', 'c>-.', 'ks-','m+-']
type2 = ['ro', 'gv', 'b<', 'c>', 'ks','m+']

label1 = ['Eq 33 n = 0.1','Eq 33 n = 0.2','Eq 33 n = 0.3','Eq 33 n = 0.4','Eq 33 n = 0.5']
label2 = ['Eq 34 n = 0.1','Eq 34 n = 0.2','Eq 34 n = 0.3','Eq 34 n = 0.4','Eq 34 n = 0.5']



for i in [0,1,2,3,4]:

    plt.plot(data['stress'][np.arange(i,50,5)]*1e6,x_n[np.arange(i,50,5)] , type1[i], label=label1[i], linewidth=2)
    plt.scatter(data['stress'][np.arange(i,50,5)]*1e6, y_n[np.arange(i,50,5)], marker= marker[i],label=label2[i], alpha=0.6, edgecolors=color[i], s=50)


plt.legend(loc='upper right', fontsize=10,ncol=2)
plt.xlabel('$\sigma_y$', fontsize=12)
plt.ylabel('$H_s/\sigma_y$', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between  Theoretical and FEM Values of $H_s/\sigma_y$', fontsize=14)



plt.subplot(122)
plt.text(-0.15, 1, subplot_labels[1], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(hs_error_fem, bins=20, edgecolor='black', alpha=0.7)
#plt.plot(fn_error_fem)
#plt.plot(ft_error_fem)
plt.axvline(np.mean(hs_error_fem), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(hs_error_fem)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(hs_error_fem)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $H_s/\sigma_y$ Error',fontsize = 14 )

plt.tight_layout()
plt.savefig('TU8.jpg',dpi = 600)
plt.show()
















