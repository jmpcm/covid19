import os
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, bisect, fmin
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from countryinfo import CountryInfo
from itertools import product
from model import *

#initial guesses for the parameters
BETA = 0.38
R0 = 3.07
R0_SIRX = 3.07
R0_SIR = 1.47
K0 = 0.01
K  = 0.01

#consider 10% quarantine (relevant only for SIR-X model)
PERCENTAGE_QUARANTINE = 0.0
# The begining of the dataseries is the day when there where START_FIT infected people
START_FIT = 100
# Consider data from START_DAYS
START_DAYS = -50
# Predict using the model until END_MODEL days
END_MODEL = 30*3

# Path to get the csv files
CSV_PATH = "COVID-19/csse_covid_19_data/csse_covid_19_daily_reports"

# Select contries from this list (othes might work as well)
COUNTRY = 'Korea, South'
COUNTRY = 'US'
COUNTRY = 'Italy'
COUNTRY = 'Brazil'
COUNTRY = 'Greece'
COUNTRY = 'Netherlands'
COUNTRY = 'United Kingdom'
COUNTRY = 'Germany'
COUNTRY = 'Austria'
COUNTRY = 'Portugal'
COUNTRY = 'Belgium'
COUNTRY = 'Spain'
COUNTRY = 'Sweden'

#compute 1 day difference
day = timedelta(days=1)

def get_country(dat,name):
    column_names = ['Country/Region','Country_Region']
    for column_name in column_names:
        if column_name in dat: break
    dataframe = dat[dat[column_name]==name]
    return dataframe

def get_country_pop(name):
    replace = {'US':'United States',
               'Korea, South': 'South Korea'}
    if name in replace:
        country = CountryInfo(replace[name])
    else:
        country = CountryInfo(name)
    return country.population()

def get_start_data_country(name,start_days=START_DAYS):
    """ Get initial data for SIR model from country"""
    pop = get_country_pop(name)
    x,y = get_data_country(name,'Infected',start_days=start_days)
    I0 = y[0]
    x,y = get_data_country(name,'Recovered',start_days=start_days)
    R0 = y[0]
    X0 = pop*PERCENTAGE_QUARANTINE
    S0 = pop-I0-R0-X0
    return [S0,I0,R0,X0]

def get_data_country(name,what,start_days=START_DAYS):
    if (what == 'Infected'):
        x,confirmed = get_data_country(name,'Confirmed',start_days=start_days)
        x,recovered = get_data_country(name,'Recovered',start_days=start_days)
        x,deaths    = get_data_country(name,'Deaths',start_days=start_days)
        return x,np.array(confirmed)-np.array(recovered)-np.array(deaths)
    x = []
    y = []
    for i in range(start_days,0):
        date = today + i*day
        filename = os.path.join(CSV_PATH,date.strftime("%m-%d-%Y")+".csv")
        dat = pd.read_csv(filename)
        dataframe = get_country(dat,name)
        x.append( i )
        y.append(dataframe[what].sum())
    return np.array(x),np.array(y)

def get_derivatives(x,y):
    dy = [0]
    for i in range(1,len(y)):
        dy.append(y[i]-y[i0])
    return x,dy

def get_model_exp(x,y):
    #fit model
    k = 0.2
    x0 = -10
    p0 = [k,x0]
    lbound = [0,-200]
    ubound = [1,0]
    args, pcov = curve_fit(fit_exp,x,y,p0,bounds=(lbound,ubound))

    #compute model
    x_prev = np.linspace(START_DAYS,END_MODEL,1000)
    y_prev = fit_exp(x_prev,*args)

    return x_prev,y_prev,args

def get_model_logistic(x,y):
    #fit model
    L = 100000
    k = 0.2
    x0 = -10
    p0 = [k,x0,L]
    lbound = [0,-200,1]
    ubound = [1,0,5000000]
    args, pcov = curve_fit(fit_logistic,x,y,p0,bounds=(lbound,ubound))

    #compute model
    x_prev = np.linspace(START_DAYS,END_MODEL)
    y_prev = fit_log(x_prev,*args)

    return x_prev,y_prev,args

def fit_logistic(x,k,x0,L):
    return L/(1+np.exp(-k*(x-x0)))

def fit_exp(x,k,x0):
    arg = k*(x-x0)
    return np.exp(arg)

def get_model_sir(country):
    # intial parameters
    N = get_country_pop(country)
    alpha = R0_SIR*BETA/N
    p0 = [alpha,BETA]
    lbound = [0,0]
    ubound = [1,1]
    print(('initial:'+'%12.4e '*len(p0))%tuple(p0))

    # get all data
    x,y = get_data_country(country,'Infected',start_days=START_DAYS)

    # Determine the day in which there were X infected
    fI = interpolate.interp1d(x,y-START_FIT)
    x0 = bisect(fI,np.min(x),np.max(x))
    ix0 = min(range(len(x)), key=lambda i: abs(x[i]-x0))
    #print('on day %d there were %d infected'%(int(x0),START_FIT))

    # Get initial conditions
    y0 = get_start_data_country(country,start_days=int(x0))
    S,I,R,X = y0
    y0 = [S+X,I,R]
    get_sir = make_get_sir(y0)
    x = x[ix0:]
    y = y[ix0:]

    # Fit
    args, pcov = curve_fit(get_sir,x,y,p0,bounds=(lbound,ubound))
    #args, pcov = curve_fit(get_sir,x,y,p0)
    print(('final:  '+'%12.4e '*len(args))%tuple(args))

    #compute model
    x_prev = np.linspace(int(x0),END_MODEL,1000)
    y_prev = get_sir(x_prev,*args)

    #error estimate
    perr = np.sqrt(np.diag(pcov))
    print(('err:    '+'%12.4e '*len(perr))%tuple(perr))
    mul_sigma = 2 # multiply by sigma
    y_err_min = np.zeros([len(y_prev),mul_sigma])
    y_err_max = np.zeros([len(y_prev),mul_sigma])
    for imul_sigma in range(mul_sigma):
        mul_fact = (imul_sigma+1)
        y_err_min[:,imul_sigma], y_err_max[:,imul_sigma] = error_estimate(args,perr*mul_fact,x_prev,y_prev,get_sir)

    return x_prev,y_prev,args,y_err_min,y_err_max

def make_get_sir(y0):
    """
    Just a way to create a function with the right signature
    for curve_fit while passing the initial arguments
    https://stackoverflow.com/questions/10250461/passing-additional-arguments-using-scipy-optimize-curve-fit/10250623
    """
    def get_sir_confirmed(x,alpha,beta):
        #densify t based on limits of y
        t = np.linspace(min(x),max(x),1000)
        sol = odeint(sir, y0, t, args=(alpha,beta))
        S,I,R = sol.T
        fI = interpolate.interp1d(t, I)
        return fI(x)
    return get_sir_confirmed

def get_model_sirx(x,y,country):
    #initial parameters
    N = get_country_pop(country)
    alpha = R0_SIRX*BETA/N
    p0 = [alpha,BETA,K,K0]
    lbound = [0,0,0,0]
    ubound = [0.5,0.5,0.5,0.5]
    print(('initial:'+'%12.4e '*len(p0))%tuple(p0))

    # get all data
    x,y = get_data_country(country,'Infected',start_days=START_DAYS)

    # Determine the day in which there were START_FIT infected
    fI = interpolate.interp1d(x,y-START_FIT)
    x0 = bisect(fI,np.min(x),np.max(x))
    ix0 = min(range(len(x)), key=lambda i: abs(x[i]-x0))
    #print('on day %d there were %d infected'%(int(x0),START_FIT))

    # Get initial conditions
    y0 = get_start_data_country(country,start_days=int(x0))
    get_sirx = make_get_sirx(y0)
    x = x[ix0:]
    y = y[ix0:]

    # Fit
    args, pcov = curve_fit(get_sirx,x,y,p0,absolute_sigma=True,bounds=(lbound,ubound))
    print(('final:  '+'%12.4e '*len(args))%tuple(args))

    #compute model
    x_prev = np.linspace(int(x0),END_MODEL)
    y_prev = get_sirx(x_prev,*args)

    #error estimate
    perr = np.sqrt(np.diag(pcov))
    print(('err:    '+'%12.4e '*len(perr))%tuple(perr))
    mul_sigma = 2 # multiply by sigma
    y_err_min = np.zeros([len(y_prev),mul_sigma])
    y_err_max = np.zeros([len(y_prev),mul_sigma])
    for imul_sigma in range(mul_sigma):
        mul_fact = (imul_sigma+1)
        y_err_min[:,imul_sigma], y_err_max[:,imul_sigma] = error_estimate(args,perr*mul_fact,x_prev,y_prev,get_sirx)

    return x_prev,y_prev,args,y_err_min,y_err_max

def error_estimate(args,perr,x_prev,y_prev,get_model):
    n = 3 # number of points in the grid
    vals = np.linspace(args-perr,args+perr,n)
    args_err = np.zeros(len(args))
    y_err_min = np.copy(y_prev)
    y_err_max = np.copy(y_prev)
    for i,idx in enumerate(product(range(n),repeat=len(args))):
        for j in range(len(args)):
            args_err[j] = vals[idx[j],j]
        y_prev_err = get_model(x_prev,*args_err)
        y_err_min = np.minimum(y_prev_err,y_err_min)
        y_err_max = np.maximum(y_prev_err,y_err_max)
    return y_err_min,y_err_max

def max_curve(x,y):
    f = interpolate.interp1d(x,-y)
    xmax = fmin(f,0,disp=False)[0]
    ymax = -f(xmax)
    return xmax,ymax

def make_get_sirx(y0):
    """
    Just a way to create a function with the right signature
    for curve_fit while passing the initial arguments
    https://stackoverflow.com/questions/10250461/passing-additional-arguments-using-scipy-optimize-curve-fit/10250623
    """
    def get_sirx_confirmed(x,alpha,beta,k,k0):
        #densify t based on limits of y
        t = np.linspace(min(x),max(x),1000)
        sol = odeint(sirx, y0, t, args=(alpha,beta,k,k0))
        S,I,R,X = sol.T
        fI = interpolate.interp1d(t, I+X)
        return fI(x)
    return get_sirx_confirmed

today = date.today()

plt.title("%s (%s) (startfit:%d)"%(COUNTRY,today.strftime("%d/%m/%Y"),START_FIT))
x,y = get_data_country(COUNTRY,'Infected')
plt.plot(x,y,'o-',label='data')

#sir
x_prev,y_prev,args,y_prev_min,y_prev_max = get_model_sir(COUNTRY)
line, = plt.plot(x_prev,y_prev,label='SIR')
try:
    xmax,ymax = max_curve(x_prev,y_prev)
    plt.axvline(xmax,c=line.get_color())
    plt.annotate("(%d,%d)"%(round(xmax),round(ymax,-3)),(xmax,max(y_prev)))
except:
    pass
_,sigma = y_prev_max.shape
#for isigma in range(sigma):
    #plt.fill_between(x_prev,y_prev_min[:,isigma],y_prev_max[:,isigma],color='grey',alpha=0.2,linewidth=0)

#sirx
x_prev,y_prev,args,y_prev_min,y_prev_max = get_model_sirx(x,y,COUNTRY)
line, = plt.plot(x_prev,y_prev,label='SIR-X')
try:
    xmax,ymax = max_curve(x_prev,y_prev)
    plt.axvline(xmax,c=line.get_color())
    plt.annotate("(%d,%d)"%(round(xmax),round(ymax,-3)),(xmax,max(y_prev)))
except:
    pass
_,sigma = y_prev_max.shape
#for isigma in range(sigma):
    #plt.fill_between(x_prev,y_prev_min[:,isigma],y_prev_max[:,isigma],color='grey',alpha=0.2,linewidth=0)

plt.axvline(0)
plt.grid()
plt.yscale('log')
plt.ylim(10,None)
plt.legend()
plt.ylabel('Number of Infected')
plt.xlabel('Days')
plt.savefig('%s.png'%COUNTRY.replace(' ','_').lower())
plt.show()

