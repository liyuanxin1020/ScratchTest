from builtins import print

from sympy import *
import numpy as np

import pandas as pd
import matplotlib as mpl
from scipy.optimize import least_squares

mpl.rcParams['font.family'] = 'Times New Roman'

def ReadCsv(filename):
    data = pd.read_csv(filename)
    #print(data.columns)
    return data

data = ReadCsv('FEM50.csv').iloc[0:50]
print(data.shape)

def solve_ss(x):

    kn1, kn = 1,1.52
    kt = 0.59
    slope_n, intercept_n, r_value_n = 0.91, 0, 0.996
    slope_t, intercept_t, r_value_t = 1.72, 0, 0.995

    n = abs(x[0])
    stress_y = abs(x[1])
    hr = hrp - hp
    strain_y = (stress_y / E)
    v = 8.1681 * pow(hrp, 3)
    vp = 2 / 3 * (w - 3.5 * (hr)) * 4.95 * hr * (hp)
    theta = 148.11 / 180 * np.pi
    c_138 = ((pow((v - vp) / (np.pi * strain_y), 1 / 3))) / (1 - np.cos(theta / 2))
    a = hrp
    # Normal force
    arg_fn = np.pi * E * strain_y * pow( np.sin(theta/2) * c_138, 3 * n) / (2 - 3 * n)
    rn1 =  hrp
    rn2 = kn * w
    fn_pre = arg_fn * (pow(rn2, 2 - 3 * n) - pow(rn1, 2 - 3 * n)) * pow(kn1, 2 - 3 * n)/np.sin(theta / 2)
    # Tangential force
    rt1 =  a
    rt2 = kt * c_138
    arg_ft1 = (n - 1) / (4 * (n + 1)) * (1 - pow(rt1 / rt2, 2))
    arg_ft2 = (pow(rt2 / rt1, 3 * n + 1) - 1) / ((3 * n + 1) * (n + 1))
    ft_pre = (arg_ft1 + arg_ft2) * pow(stress_y, 2) * pow(c_138, 2) / E * theta + u*fn_pre
    '''
    print(f'n is {n},stress_y is {stress_y * 1e-6} c_138 is {c_138 * 1e6}')
    print(f'fn_pre is {slope_n * fn_pre + intercept_n},fn is {fn}')
    print(f'ft_pre is {slope_t * ft_pre + intercept_t},ft is {ft}')
    print([fn - slope_n*fn_pre-intercept_n,ft - slope_t*ft_pre-intercept_t])
    '''


    return np.array([
        fn - slope_n * fn_pre - intercept_n,
        ft - slope_t * ft_pre - intercept_t
    ])

ss = []

for h, w, s, n, hrp, hp, ft, u, fn, E in zip(data['h']*1e-6, data['w']*1e-6, data['stress']*1e6, data['n'], data['hrp']*1e-6, data['hp']*1e-6,data['ft']*2.0, data['u'], data['fn']*2.0, data['E']*1e9):

    # Define the boundaries of variables

    bounds = ([0.1, 1e8], [0.5, 2e9])  # x  [0.1, 0.5], y  [1e8, 2e9]
    result_least_squares = least_squares(solve_ss, [n,s], bounds=bounds,ftol=0.5)
    print(f"the root is {(result_least_squares)} , the true value is {n, s}")
    ss.append([abs(result_least_squares.x), [n, s]])



print(np.array(ss).reshape(-1,4))

ss_data = np.array(ss).reshape(-1,4)

n_pre = ss_data[:,0]
s_y_pre = ss_data[:,1]*1e-6
n_true = ss_data[:,2]
s_y_true = ss_data[:,3]*1e-6

n_error = 100 * abs( n_pre- n_true) / n_true
s_y_error = 100 * abs(s_y_pre - s_y_true) / s_y_true
'''
data['n_pre'] = n_pre
data['n_error'] = n_error
data['s_y_pre'] = s_y_pre/1e6
data['s_y_error'] = s_y_error
data.to_csv('FEM-ss.csv')
'''
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,9))

subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

plt.subplot(221)

plt.text(-0.15, 1, subplot_labels[0], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.plot(n_pre, 'ro-', label='Theoretical Value', linewidth=2)
plt.scatter(np.arange(0, len(n_true)), n_true, label='FEM value', alpha=0.6, edgecolors='b', s=50)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('n', fontsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between  Theoretical and FEM Values of $n$', fontsize=14)


plt.subplot(223)
plt.text(-0.15, 1, subplot_labels[2], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(n_error, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(n_error), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(n_error)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(n_error)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $n$ Error',fontsize = 14 )



plt.subplot(222)
plt.text(-0.15, 1, subplot_labels[1], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.plot(s_y_pre, 'ro-', label='Theoretical Value', linewidth=2)
plt.scatter(np.arange(0, len(s_y_true)), s_y_true, label='FEM value', alpha=0.6, edgecolors='b', s=50)
plt.legend(loc='upper left', fontsize=10)
plt.xlabel('Data Points', fontsize=12)
plt.ylabel('$\sigma_y$(MPa)', fontsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.title('Comparison between Theoretical and FEM Values of $\sigma_y$', fontsize=14)




plt.subplot(224)
plt.text(-0.15, 1, subplot_labels[3], transform=plt.gca().transAxes, fontsize=16, fontweight='bold', va='top')

plt.hist(s_y_error, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(s_y_error), color='red', linestyle='dashed', linewidth=1)
plt.text(np.mean(s_y_error)*1.1, max(plt.ylim())*0.9, 'Mean Error: {:.2f}'.format(np.mean(s_y_error)) , fontsize = 10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Error (%)' ,fontsize = 12)
plt.ylabel('Frequency',fontsize = 12)
plt.title('Distribution of $\sigma_y$ Error',fontsize = 14 )


plt.tight_layout()
plt.savefig('TU5.jpg',dpi = 600)

plt.show()























