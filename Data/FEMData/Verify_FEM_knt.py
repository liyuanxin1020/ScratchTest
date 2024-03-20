from builtins import print
import numpy as np
from scipy.optimize import root,fsolve
import pandas as pd
import scipy
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
        he = h-hr
        stress_y = s
        strain_y = (stress_y / E)
        #print(strain_y)
        v = 8.1681 * pow(hrp, 3)
        vp = 2/3*(w-3.5*(hr))*4.95*hr*(hp)
        #print(v,vp)
        theta = 148.11/180*np.pi
        csc = ((pow( (v-vp) / (np.pi * strain_y), 1 / 3))) / (1 - np.cos(theta/2))
        #print(csc)
        a = hrp
        #print(csc,w,hrp)
        # Normal force
        arg_fn = np.pi * E * strain_y * pow(np.sin(theta/2) * csc, 3 * n)/(2-3*n)
        #print(arg_fn)
        rn1 =  hrp
        rn2 = kn * w
        fn_pre = arg_fn * (pow(rn2, 2 - 3 * n) - pow(rn1, 2 - 3 * n)) * pow(kn1, 2 - 3 * n)/np.sin(theta / 2)
        # Tangential force
        rt1 =  a
        rt2 = kt * csc
        arg_ft1 = (n - 1) / (4 * (n + 1)) *  (1 -  pow(rt1/rt2, 2) )

        arg_ft2 =  (pow(rt2 / rt1, 3 * n + 1  ) - 1)/( (3 * n + 1 ) * (n + 1))

        ft_pre = (arg_ft1 + arg_ft2)* pow(stress_y, 2) * pow(csc, 2) /E * theta + u * fn_pre

        h_e = strain_y * csc / (1 - 3 * n) + strain_y * csc / 2 - strain_y * hrp * pow(csc / hrp, 3 * n) / (1 - 3 * n)

        f_pre.append(np.array([float(fn_pre) , float(ft_pre) , fn , ft ,h_e,he]).T )

    return np.array(f_pre)




kn1,kn = 1.0,1.52

kt = 0.59

f = calculation_parameter(data,kn1,kn,kt)

x_n = np.array(f[:,0]).flatten()
y_n = np.array(f[:,2]).flatten()
x_t = np.array(f[:,1]).flatten()
y_t = np.array(f[:,3]).flatten()

slope_n, intercept_n, r_value_n = 0.91 , 0 , 0.996
slope_t, intercept_t, r_value_t = 1.72 , 0, 0.995

print(slope_n,slope_t,intercept_n,intercept_t)

fn_error_fem = 100 * abs(slope_n * f[:, 0]  - f[:, 2]) / f[:,2]
ft_error_fem = 100 * abs(slope_t * f[:, 1]   - f[:, 3]) / f[:,3]



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,9))
# 定义子图标签
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

plt.subplot(221)

plt.text(-0.15, 1, subplot_labels[0], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

# Theoretical Value
plt.plot(slope_n*x_n , 'ro-', label='Theoretical Value', linewidth=2)
#plt.scatter(np.arange(0, len(x_n)), slope_n*x_n, label='Theoretical Value', alpha=0.6, edgecolors='r', s=50)
# FEM value
plt.scatter(np.arange(0, len(y_n)), y_n, label='FEM value', alpha=0.6, edgecolors='b', s=50)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('Force(N)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between  Theoretical and FEM Values of $F_n$', fontsize=14)


plt.subplot(223)
plt.text(-0.15, 1, subplot_labels[2], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(fn_error_fem, bins=20, edgecolor='black', alpha=0.7)
#plt.plot(fn_error_fem)
#plt.plot(ft_error_fem)
plt.axvline(np.mean(fn_error_fem), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(fn_error_fem)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(fn_error_fem)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $F_n$ Error',fontsize = 14 )



plt.subplot(222)
plt.text(-0.15, 1, subplot_labels[1], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.plot(slope_t*x_t, 'ro-', label='Theoretical Value', linewidth=2)
plt.scatter(np.arange(0, len(y_t)), y_t, label='FEM value', alpha=0.6, edgecolors='b', s=50)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('Force(N)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between Theoretical and FEM Values of $F_t$', fontsize=14)


plt.subplot(224)
plt.text(-0.15, 1, subplot_labels[3], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(ft_error_fem, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(ft_error_fem), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(ft_error_fem)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(ft_error_fem)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $F_t$ Error',fontsize = 14 )


plt.tight_layout()
plt.savefig('TU4.jpg',dpi = 600)
plt.show()
















