from builtins import print
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

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
        v = 8.1681 * pow(hrp, 3)
        vp = 2/3*(w-3.5*(hr))*4.95*hr*(hp)
        theta = 148.11/180*np.pi
        c_138 = ((pow( (v-vp) / (np.pi * strain_y), 1 / 3))) / (1 - np.cos(theta/2))
        a = hrp
        arg_fn = np.pi * E * strain_y * pow(np.sin(theta/2) * c_138, 3 * n)/(2-3*n)
        rn1 =  hrp
        rn2 = kn * w
        fn_pre = arg_fn * (pow(rn2, 2 - 3 * n) - pow(rn1, 2 - 3 * n)) * pow(kn1, 2 - 3 * n)/np.sin(theta / 2)
        rt1 =  a
        rt2 = kt * c_138
        arg_ft1 = (n - 1) / (4 * (n + 1)) *  (1 -  pow(rt1/rt2, 2) )
        arg_ft2 =  (pow(rt2 / rt1, 3 * n + 1  ) - 1)/( (3 * n + 1 ) * (n + 1))
        ft_pre = (arg_ft1 + arg_ft2)* pow(stress_y, 2) * pow(c_138, 2) /E * theta + u * fn_pre
        h_e = strain_y * c_138 / (1 - 3 * n) + strain_y * c_138 / 2 - strain_y * hrp * pow(csc / hrp, 3 * n) / (1 - 3 * n)
        f_pre.append(np.array([ h_e*1e6,he*1e6]).T )

    return np.array(f_pre)


kn1,kn = 1.0,1.52

kt = 0.59

f = calculation_parameter(data,kn1,kn,kt)

x_n = np.array(f[:,0]).flatten()
y_n = np.array(f[:,1]).flatten()


he_error_fem = 100 * abs(f[:, 0]  - f[:, 1]) / f[:,1]


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,4))

subplot_labels = ['(a)', '(b)']

plt.subplot(121)

plt.text(-0.15, 1, subplot_labels[0], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')
plt.plot(x_n , 'ro-', label='Theoretical Value', linewidth=2)
plt.scatter(np.arange(0, len(y_n)), y_n, label='FEM value', alpha=0.6, edgecolors='b', s=50)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('$H_e$ ($\mu$m)', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between  Theoretical and FEM Values of $H_e$', fontsize=14)

plt.subplot(122)
plt.text(-0.15, 1, subplot_labels[1], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(he_error_fem, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(he_error_fem), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(he_error_fem)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(he_error_fem)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $H_e$ Error',fontsize = 14 )

plt.tight_layout()
plt.savefig('TU9.jpg',dpi = 600)
plt.show()
















