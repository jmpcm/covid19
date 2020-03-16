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

AVG_GROWTH_RATE_POINTS = 2
DATA_FILE_PATH = "COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
START_DAYS = -20


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


def get_country(name: str, province: str = ""):
    dataframe = dat[dat["Country/Region"] == name]

    if province:
        dataframe = dataframe[dataframe["Province/State"] == province]

    return list(dataframe.index)[0]


def get_data_country(country: str, province: str):
    today = datetime.now()
    day = timedelta(days=1)

    country_row = get_country(country, province)
    x = []
    y = []
    for i in range(START_DAYS, 0):
        date = today + i * day
        x.append(i)
        y.append(dat.at[country_row, date.strftime("%-m/%-d/%y")])
    return x, y


def get_model(x, y):
    # fit model
    L = 10000
    k = 0.3
    x0 = 0
    p0 = [k, x0]
    args, pcov = curve_fit(fit, x, y, p0, maxfev=100000)

    # compute model
    x_prev = np.linspace(START_DAYS, 7)
    y_prev = fit(x_prev, *args)

    return x_prev, y_prev


def fit_log(x, k, x0, L):
    return L / (1 + np.exp(-k * (x - x0)))


def fit(x, k, x0):
    return np.exp(k * (x - x0))


def update_repository():
    if not path.exists(DATA_FILE_PATH):
        print("Cloning repository... ", end="")
        process = subprocess.Popen(args=["git", "clone", "https://github.com/CSSEGISandData/COVID-19"],
                                   stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print(output)

    print("Pulling repository... ", end="")
    process = subprocess.Popen(args=["git", "-C COVID-19", "pull"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    print(stdout.decode("utf-8"))


if __name__ == "__main__":
    update_repository()

    # load data
    dat = pd.read_csv(DATA_FILE_PATH)
    print(dat)

    # country_row = 20  # Belgium
    # country_row = 32  # Austria
    # country_row = 11  # Germany
    # country_row = 16  # Italy
    # country_row = 60  # Portugal

    # plot data
    fig = plt.figure()

    countries = [
        # ("Austria", ""),
        # ("China", "Hubei"),
        ("Italy", ""),
        ("Denmark", ""),
        ("Spain", ""),
        ("Portugal", "")
    ]

    for i, country_province in enumerate(countries):
        x, y = get_data_country(country=country_province[0],
                                province=country_province[1])
        x_prev, y_prev = get_model(x, y)
        avg_units, avg_percentage = average_growth_rate(
            y[-AVG_GROWTH_RATE_POINTS:])
        print("Average growth rate in the last {} days in {} is {} ({:.0f}%)".format(AVG_GROWTH_RATE_POINTS,
                                                                                     country_province[0],
                                                                                     avg_units,
                                                                                     avg_percentage))

        ax = fig.add_subplot(2, 2, i+1)
        ax.plot(x_prev, y_prev, label="exp(k*(x-x0))",
                linewidth=1.5, dashes=[6, 2])
        ax.plot(x, y, linewidth=1.5, marker=".")
        for i, txt in enumerate(y):
            if txt > 0:
                ax.annotate(txt, (x[i], y[i]),
                            fontsize="x-small", rotation="90")
        ax.axvline(0)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("{}{}".format(country_province[0],
                                   "({})".format(country_province[1]) if country_province[1] else ""))
        ax.grid(True)

    plt.tight_layout()
    plt.show()
