
from builtins import print

import numpy as np

import pandas as pd

import scipy.stats as st


def ReadCsv(filename):
    data = pd.read_csv(filename)
    # print(data.columns)
    return data

data = ReadCsv('FEM12.csv')
print(data)
def calculation_parameter(data ,kn1 ,kn ,kt ):

    f_pre = []

    for h, w, s, n, hrp, hp, ft, u, fn, E in zip(data['h' ] *1e-6, data['w' ] *1e-6, data['stress' ] *1e6, data['n'], data['hrp' ] *1e-6, data['hp' ] *1e-6 ,data['ft' ]*2 , data['u'], data['fn' ]*2 , data['E' ] *1e9):

        hr = hrp - hp
        stress_y = s
        strain_y = (stress_y / E)
        # print(strain_y)
        v1 = 8.1681 * pow(hrp, 3)
        v2 = 2 / 3 * (w - 3.5 * (hr)) * 4.95 * hr * (hp)
        # print(v,vp)
        theta = 148.11 / 180 * np.pi
        c_138 = ( (pow((v1 - v2) / (np.pi * strain_y), 1 / 3)) ) / (1 - np.cos(theta / 2))

        arg_fn = np.pi * stress_y * pow(np.sin(theta / 2) * c_138, 3 * n) / (2 - 3 * n)
        #print(arg_fn)
        rn1 =  hrp
        rn2 = kn * w
        # print(rn2,rn1)
        # Normal force
        fn_pre = arg_fn * (pow(rn2, 2 - 3 * n) - pow(rn1, 2 - 3 * n)) * pow(kn1,2-3*n)/np.sin(theta / 2)

        # Tangential force
        rt1 =  hrp
        rt2 = kt * c_138
        # print(rt1,rt2)
        arg_ft1 = (n - 1) / (4 * (n + 1)) * (1 - pow(rt1 / rt2, 2))
        arg_ft2 = (pow(rt2 / rt1, 3 * n + 1) - 1) / ((3 * n + 1) * (n + 1))
        ft_pre = (arg_ft1 + arg_ft2) * pow(stress_y * c_138, 2) * theta/E

        f_pre.append(np.array([float(fn_pre), float(ft_pre), fn, ft]).T)

    return np.array(f_pre)


error = []

for kn1 in [1]:
    for kn in np.arange(0.1, 2, 0.01):
        for kt in np.arange(0.1, 2, 0.01):

                f = calculation_parameter(data, kn1, kn, kt)

                x_n = np.array(f[:, 0]).flatten()
                y_n = np.array(f[:, 2]).flatten()
                x_t = np.array(f[:, 1]).flatten()
                y_t = np.array(f[:, 3]).flatten()

                # fitting

                slope_n, intercept_n, r_value_n, p_value_n, std_err_n = st.linregress(x_n, y_n)
                slope_t, intercept_t, r_value_t, p_value_t, std_err_t = st.linregress(x_t, y_t)

                fn_error_fem = 100 * (slope_n * f[:, 0] - f[:, 2]) / f[:, 2]
                ft_error_fem = 100 * (slope_t * f[:, 1] - f[:, 3]) / f[:, 3]

                if (r_value_n > 0.99) and (r_value_t > 0.99):
                    print(f'slope_n is {slope_n:.3f} ,intercept_n is {intercept_n:.3f} \n'
                          f'slope_t is {slope_t:.3f},intercept_t is {intercept_t:.3f},\n'
                          f'fn r^2 is {r_value_n:.3f}, ft r^2 is {r_value_t:.3f} \n'
                          f'fn error is {np.mean(fn_error_fem):.3f} ft error is {np.mean(ft_error_fem):.3f} \n'
                          f'kn1 is {kn1} , kn is {kn} \n'
                          f'kt is {kt} '
                          )
                    print(fn_error_fem,ft_error_fem)

                    error.append([np.mean(abs(fn_error_fem)), np.mean(abs(ft_error_fem)), slope_n, slope_t, (intercept_n),
                                  (intercept_t), r_value_n, r_value_t, kn1, kn, kt])

error = np.array(error, dtype=object)
print(error.shape)

pd_error = pd.DataFrame(error,
                        columns=['error_n', 'error_t', 'slope_n', 'slope_t', 'intercept_n', 'intercept_t', 'r_n', 'r_t',
                                 'kn1', 'kn', 'kt'])

pd_error.to_excel('pd_error.xlsx')

pd_error_n = pd_error.iloc[:, [0, 2, 4, 6, 8, 9]]
pd_error_n.to_excel('pd_error_n.xlsx')

pd_error_t = pd_error.iloc[:, [1, 3, 5, 7, 10 ]]
pd_error_t.to_excel('pd_error_t.xlsx')

arg1 = np.argmin(abs(error[:, 4]))
arg2 = np.argmin(abs(error[:, 5]))

print(f'error_n is {error[[arg1], [0]][0]:.3f}, slope_n is {error[[arg1], [2]][0]:.3f} '
      f'intercept_n is {error[[arg1], [4]][0]:.3f} , r_n is {error[[arg1], [6]][0]:.3f} , kn1 is {error[[arg1], [8]][0]:.3f} kn is {error[[arg1], [9]][0]:.3f}')
print(f'error_t is {error[[arg2], [1]][0]:.3f}, slope_t is {error[[arg2], [3]][0]:.3f} '
      f'intercept_t is {error[[arg2], [5]][0]:.3f} , r_t is {error[[arg2], [7]][0]:.3f} , kt is {error[[arg2], [10]][0]:.3f} ')
