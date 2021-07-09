import os

import numpy as np
import pandas as pd
#import saveload as sl
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from IPython.display import clear_output

def read_inputs(data_folder):

    inp={}


    # % ********  Initial Condition  *********
    this_dir, this_filename = os.path.split(__file__)

    fn=os.path.join(this_dir, data_folder, 'initial_condition.inp')
    data=np.loadtxt(fn,delimiter=' ',usecols=[1])
    watershed_area=data[0]
    ini_values=data[1:]

    # % ********  Precipitation and Temperature  *********
    fn=os.path.join(this_dir, data_folder, 'Precipitation_Temperature.inp')
    forcing=pd.read_csv(fn,delim_whitespace=True,index_col=0,parse_dates=True,names=['P','T'])
    #forcing=pd.read_csv(fn,delim_whitespace=True,usecols=[1,2],names=['P','T'])
    #forcing.index=pd.date_range(start='1979-01-01',freq='D',periods=len(forcing))
    forcing['month_time_series']=forcing.index.month.values

    # % ********  Evapotranspiration  *********
    fn=os.path.join(this_dir, data_folder, 'monthly_data.inp')
    Tave,PEave=np.loadtxt(fn,delimiter='\t',unpack=True)
    long_term={}
    long_term['monthly_average_T']=Tave
    long_term['monthly_average_PE']=PEave

    return watershed_area, ini_values, forcing, long_term


def HBV_SASK(par_values=[], basin='banff'):

    # read parameters:
#    if par_values==[]: par_values=sl.json2dict(folder+'/pars.inp')

    # read inputs
    data_folder = basin
    [ watershed_area, ini_values, forcing, long_term ] = read_inputs(data_folder);

    Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_values,watershed_area,ini_values,forcing,long_term)

    # Pack up output fluxes
    flux=pd.DataFrame(index=forcing.index)
    flux['Q_cms'] = Q_cms
    flux['Q_mm'] = Q
    flux['AET'] = AET
    flux['PET'] = PET
    flux['Q1'] = Q1
    flux['Q1routed'] = Q1_routed
    flux['Q2'] = Q2
    flux['ponding'] = ponding

    # Pack up output states
    state=pd.DataFrame(index=pd.date_range(start=flux.index[0],periods=len(SWE),freq='D'))
    state['SWE'] = SWE
    state['SMS'] = SMS
    state['S1'] = S1
    state['S2'] = S2

    return flux, state, forcing

def run_model(par_values,watershed_area,ini_values,forcing,long_term):
    # Unpack parameters:
    TT = par_values['TT']
    C0 = par_values['C0']
    ETF = par_values['ETF']
    LP = par_values['LP']
    FC = par_values['FC']
    beta = par_values['beta']
    FRAC = par_values['FRAC']
    K1 = par_values['K1']
    alpha = par_values['alpha']
    K2 = par_values['K2']

    if par_values['UBAS'] == []:
        UBAS=1
    else:
        UBAS = par_values['UBAS']

    if par_values['PM'] == []:
        PM=1
    else:
        PM=par_values['PM']

    LP = LP * FC;

    # Unpack initial conditions and forcing
    initial_SWE = ini_values[0]; initial_SMS = ini_values[1];
    initial_S1= ini_values[2];   initial_S2 = ini_values[3];

    P = PM * forcing['P'].values
    T = forcing['T'].values
    month_time_series = forcing['month_time_series'].values
    monthly_average_T = long_term['monthly_average_T']
    monthly_average_PE = long_term['monthly_average_PE']

    period_length = len(P)

    SWE=np.zeros(period_length+1)
    SWE[0] = initial_SWE;

    SMS=np.zeros(period_length+1)
    SMS[0] = initial_SMS;

    S1=np.zeros(period_length+1)
    S1[0] = initial_S1

    S2=np.zeros(period_length+1)
    S2[0] = initial_S2


    ponding=np.zeros(period_length)
    AET=np.zeros(period_length)
    PET=np.zeros(period_length)
    Q1=np.zeros(period_length)
    Q2=np.zeros(period_length)

    for t in range(period_length):

        SWE[t+1],ponding[t]=precipitation_module(SWE[t],P[t],T[t],TT,C0)

        AET[t],PET[t]=evapotranspiration_module(SMS[t],T[t],month_time_series[t],monthly_average_T,monthly_average_PE,ETF,LP)

        SMS[t+1],S1[t+1],S2[t+1],Q1[t],Q2[t]=soil_storage_routing_module(ponding[t], SMS[t],
                                                S1[t], S2[t], AET[t],
                                                FC, beta, FRAC, K1, alpha, K2)

    Q1_routed = triangle_routing(Q1, UBAS)
    Q = Q1_routed + Q2
    Q_cms=(Q*watershed_area*1000)/(24*3600)

    return Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2

def precipitation_module( SWE, P, T, TT, C0):
# % *****  TT : Temperature Threshold or melting/freezing point - model parameter *****
# % *****  C0: base melt factor - model parameter *****
# % *****  P: Precipitation - model forcing *****
# % *****  T: Temperature - model forcing *****
# % *****  SWE: Snow Water Equivalent - model state variable *****
    if T >= TT:
        rainfall = P
        potential_snow_melt  = C0 * (T - TT)
        snow_melt = min((potential_snow_melt,SWE))
        ponding = rainfall + snow_melt # Liquid Water on Surface
        SWE_new = SWE - snow_melt # Soil Water Equivalent - Solid Water on Surface
    else:
        snowfall = P
        snow_melt = 0
        ponding = 0 # Liquid Water on Surface
        SWE_new = SWE + snowfall # Soil Water Equivalent - Solid Water on Surface

    return SWE_new, ponding

def evapotranspiration_module(SMS,T,month_number, monthly_average_T,monthly_average_PE,ETF,LP):
# % *****  T: Temperature - model forcing *****
# % *****  month_number: the current month number - for Jan=1, ..., Dec=12 *****
# % *****  SMS: Soil Moisture Storage - model state variable *****
# % *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters
# % *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter
# % *****  PET: Potential EvapoTranspiration - model parameter
# % *****  AET: Actual EvapoTranspiration - model

    # Potential Evapotranspiration:
    PET = ( 1 + ETF * ( T - monthly_average_T[month_number-1] ) ) * monthly_average_PE[month_number-1]
    PET = max((PET, 0))

    if SMS > LP:
        AET = PET
    else:
        AET = PET * ( SMS / LP )

    AET = min((AET, SMS)) # to avoid evaporating more than water available

    return AET,PET

def soil_storage_routing_module(ponding, SMS, S1, S2, AET, FC, beta, FRAC, K1, alpha, K2):
#     % *****  T: Temperature - model forcing *****
#     % *****  month_number: the current month number - for Jan=1, ..., Dec=12 *****
#     % *****  SMS: Soil Moisture Storage - model state variable *****
#     % *****  ETF - This is the temperature anomaly correction of potential evapotranspiration - model parameters
#     % *****  LP: This is the soil moisture content below which evaporation becomes supply-limited - model parameter
#     % *****  PET: Potential EvapoTranspiration - model parameter


#     % *****  FC: Field Capacity - model parameter ---------
#     % *****  beta: Shape Parameter/Exponent - model parameter ---------
#     % This controls the relationship between soil infiltration and soil water release.
#     % The default value is 1. Values less than this indicate a delayed response, while higher
#     % values indicate that runoff will exceed infiltration.

    if SMS < FC:
        soil_release = ponding * (( SMS / FC )**beta) # release of water from soil
    else:
        soil_release = ponding # release of water from soil

    SMS_new = SMS - AET + ponding - soil_release

#     % if SMS_new < 0 % this might happen due to very small numerical/rounding errors
#     %     SMS_new
#     %     SMS_new = 0;
#     % end

    soil_release_to_fast_reservoir = FRAC * soil_release
    soil_release_to_slow_reservoir = ( 1 - FRAC ) * soil_release

    Q1 = K1*S1**alpha
    if Q1>S1:
        Q1=S1

    S1_new = S1 + soil_release_to_fast_reservoir - Q1

    Q2 = K2 * S2

    S2_new = S2 + soil_release_to_slow_reservoir - Q2

    return SMS_new, S1_new, S2_new, Q1, Q2

def triangle_routing(Q, UBAS):
    UBAS = max((UBAS, 0.1))
    length_triangle_base = int(np.ceil(UBAS))
    if UBAS == length_triangle_base:
        x = np.array([0, 0.5*UBAS, length_triangle_base])
        v = np.array([0, 1, 0])
    else:
        x = np.array([0, 0.5*UBAS, UBAS, length_triangle_base])
        v = np.array([0, 1, 0, 0])
    
    weight=np.zeros(length_triangle_base)

    for i in range(1,length_triangle_base+1):
        if (i-1) < (0.5 * UBAS) and i > (0.5 * UBAS):
            weight[i-1] = 0.5 * (np.interp(i - 1,x,v) + np.interp(0.5 * UBAS,x,v) ) * ( 0.5 * UBAS - i + 1) +  0.5 * ( np.interp(0.5 * UBAS,x,v) + np.interp(i,x,v) ) * ( i - 0.5 * UBAS )
        elif i > UBAS:
            weight[i-1] = 0.5 * ( np.interp(i-1,x,v) ) * ( UBAS - i + 1)
        else:
            weight[i-1] = np.interp(i-0.5,x,v)

    weight = weight/np.sum(weight)

    Q_routed=np.zeros(len(Q))
    for i in range(1,len(Q)+1):
        temp = 0
        for j in range(1,1+min(( i, length_triangle_base))):
            temp = temp + weight[j-1] * Q[i - j]
        Q_routed[i-1] = temp
    return Q_routed

def obs_streamflow(folder):

    fn=folder + '/streamflow.inp'
    Qobs=pd.read_csv(fn,delim_whitespace=True,index_col=0,parse_dates=True,names=['Q'])
    return Qobs
 
def run_optimization(folder,dates,metric,par_bounds,par_values,pn,pv):

    # read inputs
    data_folder = par_values['basin']
    [ watershed_area, ini_values, forcing, long_term ] = read_inputs(data_folder);

    # read observed streamflow
    Qobs=obs_streamflow(par_values['basin'])

    # Truncate data to calibration period only, including spinup period
    forcing=forcing[dates['start_spin']:dates['end_calib']]
    Qobs=Qobs[dates['start_calib']:dates['end_calib']]

    # Truncate par_bounds to only those parameters being optimized:
    pb=tuple([par_bounds[i] for i in pn])
    # Run optimization
    print(pv)
    print(pn)
    output=minimize(error_fun,pv,args=(pn,par_values,metric,Qobs,watershed_area, ini_values, forcing, long_term,dates),bounds=pb)
    pv=output['x']
    for n,v in zip(pn,pv): par_values[n]=v
    print(par_values)
    return par_values, pv

def error_fun(pv,pn,par_values,metric,Qobs,watershed_area, ini_values, forcing, long_term,dates):
    
    for n,v in zip(pn,pv): par_values[n]=v
    Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_values,watershed_area,ini_values,forcing,long_term)
    Q=pd.DataFrame(index=forcing.index)
    Q['Q']=Q_cms
    Q=Q[dates['start_calib']:dates['end_calib']].values
    err=eval_metric(Qobs.values,Q,metric)

    # Diplay current parameter values and objective function value:
    clear_output(wait = True)
    for n,v in zip(pn,pv): print('%s: %f'%(n,v))
    print('%s: %.4f\n'%(metric,err))

    return err

def eval_metric(yobs,y,metric):
    # Make sure the data sent in here are not in dataframes

    if metric.upper() == 'NSE':
        # Use negative NSE for minimization
        denominator = ((yobs-yobs.mean())**2).mean()
        numerator = ((yobs - y)**2).mean()
        negativeNSE = -1*(1 - numerator / denominator)
        return negativeNSE

    elif metric.upper() == 'NSE_LOG':
        # Use negative NSE for minimization
        yobs=np.log(yobs)
        y=np.log(y)
        denominator = ((yobs-yobs.mean())**2).mean()
        numerator = ((yobs - y)**2).mean()
        negativeNSE = -1*(1 - numerator / denominator)
        return negativeNSE

    elif metric.upper() == 'ABSBIAS':
        return np.abs((y-yobs).sum()/yobs.sum())

    elif metric.upper() == 'ME':
        return (yobs-y).mean()

    elif metric.upper() == 'MAE':
        return (np.abs(yobs-y)).mean()

    elif metric.upper() == 'MSE':
        return ((yobs-y)**2).mean()

    elif metric.upper() == 'RMSE':
        return np.sqrt(((yobs-y)**2).mean())

def MonteCarlo(nReal,pn,pv,par_values,par_bounds,Qobs,dates):
    par_array=np.random.rand(nReal,len(pn))
    for i,n in enumerate(pn):
        lowerlimit,upperlimit=par_bounds[n]
        par_array[:,i]=par_array[:,i]*(upperlimit-lowerlimit)+lowerlimit

    NSE=np.zeros(nReal)
    RMSE=np.zeros(nReal)

    watershed_area, ini_values, forcing, long_term = read_inputs(par_values['basin']);

    # read observed streamflow
    Qobs=obs_streamflow(par_values['basin'])

    # Truncate data to calibration period only, including spinup period
    forcing=forcing[dates['start_spin']:dates['end_calib']]
    Qobs=Qobs[dates['start_calib']:dates['end_calib']]

    for i in range(nReal):
        pv=par_array[i,:]
        for n,v in zip(pn,pv): par_values[n]=v
        
        #flux,state,forcing=HBV_SASK(par_values['basin'],par_values)
        
        Q_cms,Q,AET,PET,Q1,Q2,Q1_routed,ponding,SWE,SMS,S1,S2=run_model(par_values,watershed_area,ini_values,forcing,long_term)
        flux=pd.DataFrame(index=forcing.index)
        flux['Q_cms'] = Q_cms
        
        NSE[i]=-eval_metric(Qobs[dates['start_calib']:dates['end_calib']].values.squeeze(),flux['Q_cms'][dates['start_calib']:dates['end_calib']].values,'NSE')
        RMSE[i]=eval_metric(Qobs[dates['start_calib']:dates['end_calib']].values.squeeze(),flux['Q_cms'][dates['start_calib']:dates['end_calib']].values,'RMSE')
        clear_output(wait = True)
        print('Running realization %d'%(i+1))

    return par_array,NSE,RMSE

def dottyplots(par_array,metric,metric_name,pn,behavioural_threshold=[]):
    n=len(pn)
    plotdims=[(1,1,4,4),(1,2,8,4),(1,3,12,4),
             (2,2,8,8),(2,3,12,8),(2,3,12,8),
             (3,3,12,12),(3,3,12,12),(3,3,12,12),
             (4,3,12,16),(4,3,12,16),(4,3,12,16)]
    pa,pb,pc,pd=plotdims[n-1]

    pl.figure(figsize=(pc,pd))
    for i in range(n):
        pl.subplot(pa,pb,i+1)
        pl.plot(par_array[:,i],metric,'.k')
        if behavioural_threshold!=[]:
            xl=pl.gca().get_xlim()
            pl.plot(xl,[behavioural_threshold,behavioural_threshold],'-b')
            pl.plot(par_array[metric<=behavioural_threshold,i],metric[metric<=behavioural_threshold],'or')
        pl.xlabel(pn[i],fontsize=13)
        if i/pb==int(i/pb): pl.ylabel(metric_name,fontsize=13)
        pl.grid()

def GLUE(metric,pn,par_array,par_values,behavioural_threshold):

    par_array=par_array[metric<=behavioural_threshold,:]
    nB=par_array.shape[0]
    
    Q=pd.DataFrame()
    AET=pd.DataFrame()
    for i in range(nB):
        pv=par_array[i,:]
        for n,v in zip(pn,pv): par_values[n]=v
        flux,state,forcing=HBV_SASK(par_values['basin'],par_values)
        Q[i]=flux['Q_cms']
        AET[i]=flux['AET']
    
    # Get range of flow and AET:
    Qrange=pd.DataFrame()
    Qrange['max']=Q.max(axis=1)
    Qrange['min']=Q.min(axis=1)
    Qrange['med']=Q.median(axis=1)
    Qrange['mean']=Q.mean(axis=1)
    
    AETrange=pd.DataFrame()
    AETrange['max']=AET.max(axis=1)
    AETrange['min']=AET.min(axis=1)
    AETrange['med']=AET.median(axis=1)
    AETrange['mean']=AET.mean(axis=1)
    
    return Q,Qrange,AET,AETrange

# Calculate Change in storage:
def DeltaS(state,start,end):
    S=np.zeros(len(state['SWE']))
    for n in state: S=S+state[n]
    S=S[start:end]
    S=S-S[0]
    return S

def WaterBalancePlot(flux,state,forcing,start,end):
    # Do a nice water balance plot
    t=flux['PET'][start:end].index
    S=DeltaS(state,start,end)
    P=forcing['P'][start:end].cumsum()
    AET=flux['AET'][start:end].cumsum()
    Q=flux['Q_mm'][start:end].cumsum()

    pl.figure(figsize=(10,5))
    pl.fill_between(t,Q+AET+S,0., color='darkgreen',label='cumulative Q')
    pl.fill_between(t,S+AET,0., color='forestgreen',label='cumulative AET')
    pl.fill_between(t,S,0., color='lightgreen',label='$\Delta S$')
    pl.plot(forcing['P'][start:end].cumsum(),label='cumulative P',color='b')
    # pl.fill_between(flux['Q_mm'][start:end].cumsum()+flux['AET'][start:end].cumsum()+S,label='Streamflow',color='r')
    # pl.plot(flux['AET'][start:end].cumsum(),label='AET',color='darkgreen')

    pl.legend(fontsize=13)
    pl.ylabel('Water balance (mm)',fontsize=13); pl.grid()
    
def PlotEverything(flux,state,forcing,start,end,freq):
    # Do a nice plot of model outputs:
    tS=state['SWE'].resample(freq).mean()[start:end].index
    SWE=(state['SWE'].resample(freq).mean())[start:end]
    SMS=(state['SMS'].resample(freq).mean())[start:end]
    S1=(state['S1'].resample(freq).mean())[start:end]
    S2=(state['S2'].resample(freq).mean())[start:end]

    t=flux['PET'][start:end].resample(freq).sum().index
    P=forcing['P'][start:end].resample(freq).sum()
    AET=flux['AET'][start:end].resample(freq).sum()
    PET=flux['PET'][start:end].resample(freq).sum()
    Q=flux['Q_mm'][start:end].resample(freq).sum()


    pl.figure(figsize=(10,7))
    pl.subplot(2,1,1)
    pl.fill_between(t,PET,0., color='lightgreen',label='PET',step='pre')
    pl.step(t,P,label='P',color='b')
    pl.step(t,Q,label='Q',color='m')
    pl.step(t,AET,label='AET',color='g')
    pl.legend(fontsize=13)
    pl.ylabel('Fluxes (mm)',fontsize=13); pl.grid()

    pl.subplot(2,1,2)
    pl.step(tS,SWE,label='SWE',color='tab:cyan')
    pl.step(tS,SMS,label='SMS',color='tab:grey')
    pl.step(tS,S1,label='S1',color='tab:olive')
    pl.step(tS,S2,label='S2',color='tab:brown')
    pl.legend(fontsize=13)
    pl.ylabel('States (mm)',fontsize=13); pl.grid()

