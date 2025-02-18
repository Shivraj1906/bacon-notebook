import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

def calibrate(age, error, cc=1, eps=1e-5):
    # load calibration curve
    # cc=1: IntCal20
    # cc=2: Marine20
    if cc == 1:
        cal_curve = pd.read_csv('calib_curves/intcal20.csv')
        # extract data from calibration curve.
        # reversing the order to make it increasing.
        # which is required for interpolation by numpy.
        cal_ages = cal_curve['CAL BP'].values[::-1]
        c14_ages = cal_curve['14C age'].values[::-1]
        sigma = cal_curve['Sigma'].values[::-1]
    else:
        print('Other calibration curves are not implemented yet...')
    
    # prepare the age grid
    age_grid = np.arange(cal_ages.min(), cal_ages.max() + 1)

    # interpolate the calibration curve
    mu = np.round(np.interp(age_grid, cal_ages, c14_ages))
    tau1 = np.round(np.interp(age_grid, cal_ages, sigma))

    # calulating total error
    tau = error**2 + tau1**2

    # calculating probabilities
    probs = stats.norm.pdf((mu - age) / np.sqrt(tau))
    # normalizing probabilities to 1
    probs = probs / probs.sum()
    
    # preparing resultant dataframe
    result = pd.DataFrame(data={
        'ages': age_grid[::-1],
        'probs': probs[::-1]
    })

    # truncating probs which are less tha eps
    result = result[result['probs'] > eps]
    return result.reset_index(drop=True)