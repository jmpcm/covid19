# Authors: Jorge Miranda and Henrique Miranda

from datetime import datetime, timedelta
from os import path
import subprocess
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

AVG_GROWTH_RATE_POINTS = 3
DATA_FILE_PATH = "COVID-19/csse_covid_19_data/csse_covid_19_daily_reports"
START_DAYS = -20
WHAT = 'Deaths'
WHAT = 'Confirmed'

def average_growth_rate(values) -> Tuple[int, int]:
    """Returns the average growth rate, in units and percentage."""
    n_points = len(values) - 1

    if n_points < 1:
        raise ValueError(
            "The number of points for calculating the average is {}".format(n_points))

    sum_units = 0
    sum_percentage = 0

    for i in range(n_points):
        value = values[i + 1] - values[i]
        sum_units += value
        sum_percentage += (value / values[i + 1])

    return (sum_units/n_points, round(sum_percentage/n_points * 100, 0))


def get_country(dat,country):
    column_names = ['Country/Region','Country_Region']
    for column_name in column_names:
        if column_name in dat: break
    dataframe = dat[dat[column_name]==country]
    return list(dataframe.index)[0]

def get_data_country(country):
    today = datetime.now()
    day = timedelta(days=1)

    x = []
    y = []
    for i in range(START_DAYS,0):
        date = today + i*day
        filename = date.strftime("%m-%d-%Y")+".csv"
        dat = pd.read_csv(path.join(DATA_FILE_PATH,filename))
        country_row = get_country(dat,country)
        x.append( i )
        y.append( dat.at[country_row,WHAT] )
    return x,y

def get_model_logistic(x, y):
    # fit model
    L = 100000
    k = 0.3
    x0 = -10
    p0 = [k, x0, L]
    lbound = [0,-200,1]
    ubound = [1,0,5000000]
    args, pcov = curve_fit(fit_logistic, x, y, p0, bounds=(lbound,ubound), maxfev=100000)

    # compute model
    x_prev = np.linspace(START_DAYS, 7)
    y_prev = fit_logistic(x_prev, *args)

    return x_prev, y_prev, args

def get_model_exp(x, y):
    # fit model
    k = 0.3
    x0 = -10
    p0 = [k, x0]
    lbound = [0,-200]
    ubound = [1,0]
    args, pcov = curve_fit(fit_exp, x, y, p0, bounds=(lbound,ubound), maxfev=100000)

    # compute model
    x_prev = np.linspace(START_DAYS, 7)
    y_prev = fit_exp(x_prev, *args)

    return x_prev, y_prev, args


def fit_logistic(x, k, x0, L):
    return L / (1 + np.exp(-k * (x - x0)))


def fit_exp(x, k, x0):
    return np.exp(k * (x - x0))


def update_repository():
    if not path.exists(DATA_FILE_PATH):
        print("Cloning repository... ", end="")
        subprocess.call(args=["git", "clone", "https://github.com/CSSEGISandData/COVID-19"],
                        stdout=subprocess.PIPE)

    print("Pulling repository... ", end="")
    subprocess.call(args=["git", "-C", "COVID-19", "pull"])


if __name__ == "__main__":
    update_repository()

    # plot data
    fig = plt.figure()

    countries = [
        ("Austria", ""),
        # ("China", "Hubei"),
        ("Italy", ""),
        # ("Vietnam", ""),
        #("Denmark", ""),
        ("Belgium", ""),
        ("Spain", ""),
        ("Portugal", "")
    ]

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(WHAT)
    for i, country_province in enumerate(countries):
        x, y = get_data_country(country=country_province[0])
        line, = ax.plot(x, y, linewidth=1.5, marker=".")
        avg_units, avg_percentage = average_growth_rate(y[-AVG_GROWTH_RATE_POINTS:])
        print("Average growth rate in the last {} days in {} is {} ({:.0f}%)".format(AVG_GROWTH_RATE_POINTS,
                                                                                     country_province[0],
                                                                                     avg_units,
                                                                                     avg_percentage))


        x_prev, y_prev, args = get_model_exp(x, y)
        label_exp = 'exp(%6.4lf*(x%+lf)) %s'%(args[0],args[1],country_province[0])
        ax.plot(x_prev, y_prev, label=label_exp, linewidth=1.5, dashes=[6, 2], color=line.get_color())

        x_prev, y_prev, args = get_model_logistic(x, y)
        label_log = '%8d / exp(%6.4lf*(x%+6.4lf)) %s'%(args[2],args[0],args[1],country_province[0])
        label_log = None
        ax.plot(x_prev, y_prev, label=label_log, linewidth=1.5, dashes=[6, 2], color=line.get_color())
        ax.annotate("%s:%d"%(country_province[0],round(args[2],-3)), (x_prev[-1], args[2]),fontsize="x-small")
        ax.axhline(args[2],lw=1,alpha=0.3,color=line.get_color())

        for i, txt in enumerate(y):
            if txt < 0: continue
            ax.annotate(txt, (x[i], y[i]),fontsize="x-small", rotation="90")
        ax.axvline(0)

        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    plt.xlim(None,x_prev[-1]+5)
    plt.legend()
    plt.tight_layout()
    plt.show()
