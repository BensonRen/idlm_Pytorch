# -*- coding: utf-8 -*-
"""Coronavirus SIR Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C21mYs8tIF27MZSjOkXEMhmfdM0LzauO

# Get Data

Data sourced from [this GitHub repository](https://github.com/CSSEGISandData/COVID-19) maintained by JHU Center for Systems Science and Engineering.
"""

import pandas as pd
import numpy as np
import io
from datetime import timedelta, date

# get the dates in between two dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

base_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
start_date = date(2020,1,22)
end_date = date.today()

# save the csv files and create a time index
raw_data = []
time_index = []

for day in daterange(start_date,end_date):
    url = base_url + day.strftime("%m-%d-%Y")+".csv"
    raw_data.append(pd.read_csv(url))
    time_index.append(day)

# create three dataframe
columns = ['Confirmed','Deaths','Recovered']

US_df = pd.DataFrame(index=time_index, columns=columns)
NY_df = pd.DataFrame(index=time_index, columns=columns)
NYC_df = pd.DataFrame(index=time_index, columns=columns)

for i in range(len(raw_data)):
  
  # get US data
  try:
    US_temp = raw_data[i][raw_data[i]['Country/Region']=='US']
  except:
    US_temp = raw_data[i][raw_data[i]['Country_Region']=='US']
  US_df.iloc[i]['Confirmed'] = US_temp['Confirmed'].sum()
  US_df.iloc[i]['Deaths'] = US_temp['Deaths'].sum()
  US_df.iloc[i]['Recovered'] = US_temp['Recovered'].sum()

  NY_Confirmed = 0
  NY_Deaths = 0
  NY_Recovered = 0

  # get NY state data
  for j in range(len(US_temp)):
    try:
      if 'NY' in US_temp.iloc[j]['Province/State'] or (US_temp.iloc[j]['Province/State'] == 'New York'):
        NY_Confirmed += US_temp.iloc[j]['Confirmed']
        NY_Deaths += US_temp.iloc[j]['Deaths']
        NY_Recovered += US_temp.iloc[j]['Recovered']
    except:
      NY_temp = US_temp[US_temp['Province_State'] == 'New York']
      NY_Confirmed = NY_temp['Confirmed'].sum()
      NY_Deaths = NY_temp['Deaths'].sum()
      NY_Recovered = NY_temp['Recovered'].sum()
      
  NY_df.iloc[i]['Confirmed'] = NY_Confirmed
  NY_df.iloc[i]['Deaths'] = NY_Deaths
  NY_df.iloc[i]['Recovered'] = NY_Recovered

  # get NYC data (data incomplete)
  try:
    NYC_df.iloc[i]['Confirmed'] = US_temp[US_temp['Admin2']=='New York City']['Confirmed'].sum()
    NYC_df.iloc[i]['Deaths'] = US_temp[US_temp['Admin2']=='New York City']['Deaths'].sum()
    NYC_df.iloc[i]['Recovered'] = US_temp[US_temp['Admin2']=='New York City']['Recovered'].sum()
  except:
    NYC_df.iloc[i]['Confirmed'] = 0
    NYC_df.iloc[i]['Deaths'] = 0
    NYC_df.iloc[i]['Recovered'] = 0

# US_df.to_csv('US_data.csv')
# NY_df.to_csv('NY_data.csv')
# NYC_df.to_csv('NYC_data.csv')



"""# Basic Analysis / Visualization"""

import seaborn as sns
import matplotlib.pyplot as plt

US_df['New Confirmed'] = US_df['Confirmed'].diff()
US_df['New Deaths'] = US_df['Deaths'].diff()
US_df['New Recovered'] = US_df['Recovered'].diff()

NY_df['New Deaths'] = NY_df['Deaths'].diff()
NY_df['New Confirmed'] = NY_df['Confirmed'].diff()
NY_df['New Recovered'] = NY_df['Recovered'].diff()

US_exNY_df = US_df - NY_df

# drop the incomplete data 
# usually don't need this step
# US_df = US_df[:-1]
# US_exNY_df = US_exNY_df[:-1]
# NY_df = NY_df[:-1]

t = time_index

"""# Calibrate Model Parameters"""


"""## Try to Calibrate the Parameters"""

# to solve an initial value problem for ODEs
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class SIR(object):
  def __init__(self, data, Loss, pred_lenth, S_0, I_0,R_0):
    self.data = data
    self.Loss = Loss
    self.pred_lenth = pred_lenth
    self.S_0 = S_0
    self.I_0 = I_0
    self.R_0 = R_0

  def predict(self, pred_lenth, beta, gamma, infected, deaths, recovered):
    T = data.index.values
    current = data.index[-1]

    # extend the index
    for i in range(pred_lenth):
      current += timedelta(days=1)
      T = np.append(T, current)
    size = len(data) + pred_lenth

    def model(t, y):
      S = y[0]
      I = y[1]
      R = y[2]
      dSdt = -beta*S*I
      dIdt = beta*S*I-gamma*I
      dRdt = gamma*I
      return [dSdt, dIdt, dRdt]

    pred = solve_ivp(model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))
    extended_infected = np.concatenate((infected.values, [None]*pred_lenth))
    extended_deaths = np.concatenate((deaths.values, [None]*pred_lenth))
    extended_recovered = np.concatenate((recovered.values, [None]*pred_lenth))
    return T, extended_infected, extended_deaths, extended_recovered, pred
    
  def train(self):
    data= self.data
    recovered = data['Recovered']
    deaths = data['Deaths']
    infected = data['Confirmed'] - recovered - deaths
    optimization = minimize(Loss, [0.001,0.001], args=(infected, recovered, self.S_0, self.I_0, self.R_0),method='L-BFGS-B',bounds=[(0.0000001,0.4),(0.0000001, 0.4)])
    beta, gamma = optimization.x
    print(optimization)
    
    T, extended_infected, extended_deaths, extended_recovered, pred = self.predict(beta, gamma, pred_lenth, beta, gamma, infected, deaths, recovered, self.S_0, self.I_0, self.R_0)
    
    result_df = pd.DataFrame({'Real Infected':extended_infected, 'Real Deaths': extended_deaths, 'Real Recovered': extended_recovered, 'Predict Susceptible': prediction.y[0], 'Predict Infected': prediction.y[1], 'Predict Recovered': prediction.y[2]}, index = T)
    fig, ax = plt.subplots(figsize=(15, 10))
    df.plot(ax=ax)
    print(f"beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
  
def Loss(rates, infected, recovered, S_0, I_0,R_0):
  size = len(infected)
  beta, gamma = rates
  def model(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    dSdt = -beta*S*I
    dIdt = beta*S*I-gamma*I
    dRdt = gamma*I
    return [dSdt, dIdt, dRdt]
  # solve the system
  rez = solve_ivp(model, [0,size], [S_0, I_0, R_0], t_eval=np.arange(0,size,1), vectorized = True)
  # calculate the loss
  loss_i = np.sqrt(np.mean((rez.y[1]-infected)**2))
  loss_r = np.sqrt(np.mean((rez.y[2]-recovered)**2))
  return np.mean(loss_i+loss_r)

N_NY = 8623000
I_NY = 10
R_NY = 0

N_US = 327200000 - N_NY
I_US = 300 - I_NY
R_US = 0

pred_lenth = 90

us_sir = SIR(US_exNY_df,Loss,30,N_US,I_US,R_US)
us_sir.train()
