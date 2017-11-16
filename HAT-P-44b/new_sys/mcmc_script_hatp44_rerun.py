#!home/usr/miniconda3/envs/astroconda3/python #laptop
#!home/usr/miniconda2/envs/astroconda/python #Tamura server
'''
0. tmux new -s hatp44b
1. source activate astroconda/3
2. python mcmc_sript_hatp44.py

Note: 
This script was cloned from mcmc_script_hatp44_rerun.py

Improvements:
* systematics: flux / transit_model
* chi2: (flux / sys_model - transit_model)/err)^2 
* err is inflated by reduced chi2 and beta factor
* uniform prior on rho star is added
'''
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import getpass
from tqdm import tqdm
from astropy import units
from astropy import constants

from pytransit import MandelAgol
import scipy.optimize as op
import limbdark as ld
from scipy import stats

import emcee
from tqdm import tqdm
import gzip

#styling
pl.style.use('seaborn-white')
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
fontsize=18
pl.rcParams['ytick.labelsize'] = 'large'
pl.rcParams['xtick.labelsize'] = 'large'

environ = 'astroconda'
if os.environ['CONDA_DEFAULT_ENV'] == environ:
    pass
else:
    print('{} not found'.format(environ))
    sys.exit(0)


##############################################
#
# Prepare data
#
##############################################

data_dir = '.'
file_list=glob.glob(data_dir+'/*.csv')
file_list.sort()
len(file_list)

data_dir='.'

name='hatp44'
date='170215'
target_star_id='2'
comparison_star_id='13'#3
radii_range='9-14'

data={}
bands='g,r,z'.split(',')
sigma=5

for b in bands:
    fname='lcf_msct_'+b+'_'+name+'_'+date+'_t'+target_star_id+'_c'+comparison_star_id+'_r'+radii_range+'.bjd.dat'
    df=pd.read_csv(os.path.join(data_dir,fname), delimiter=' ', parse_dates=True)
    #manipulate columns
    cols = df.columns.tolist()
    cols.remove('#')
    cols.insert(-1,' ')
    df.columns = cols
    try:
        df=df.drop(['Unnamed: 21','frame', ' '],axis=1)
    except:
        pass
    df=df.set_index('BJD(TDB)-2450000')
    #df['BJD(TDB)'] = df['BJD(TDB)-2450000'].apply(lambda x: x + 2450000)
    #df=df.set_index('BJD(TDB)')
    #remove outliers
    df=df[np.abs(df-df.mean())<=(sigma*df.std())]
    data[b]=df.dropna()


##############################################
#
# Plot raw data with uncertainties
#
##############################################

fluxcol =  'flux(r=11.0)'
errcol  =  'err(r=11.0)'

colors='b,g,r'.split(',')
fig,ax = pl.subplots(1,1,figsize=(15,4))

n=0
for b,c in zip(bands,colors):
    df = data[b]
    offset = n*0.02
    
    time = df.index
    flux = df[fluxcol]
    err  = df[errcol]
    
    #print(len(df))
    
    ax.errorbar(time, flux-offset, yerr=err, label=b, color=c, alpha=0.5)
    ax.set_ylabel('Normalized flux', fontsize=fontsize)
    n+=1
pl.legend(loc='lower left', fontsize=fontsize)
pl.xlabel('BJD(TDB)-2450000', fontsize=fontsize)
fig.savefig('raw_data')


##############################################
#
# Known parameters
#
##############################################

_tc  = 2455696.93695
_P   = 4.301219
_inc = np.deg2rad(89.10)
_t14  = 0.13020
_b    = 0.172
Rp = 1.24 #Rjup
Rs = 0.949*units.Rsun.to(units.Rjup) #Rsol to Rjup
k_ = Rp/Rs
_a_s = 11.52
#a_s_  = scaled_a(_P, _t14, k_, i=_inc, impact_param=_b)
tc_0      = 7.8e3+0.22 #-2450000

##############################################
#
# light curve modeling functions
#
##############################################

def scaled_a(p, t14, k, i=np.pi/2, impact_param=0):
    """
    Winn 2014 ("Transits and Occultations"), eq. 14
    """
    numer = np.sqrt( (k + 1)**2 - impact_param**2 )
    denom = np.sin(i) * np.sin(t14 * np.pi / p)
    return float(numer / denom)

def u_to_q(u1, u2):
    '''
    convert limb-darkening coefficients
    from u to q
    
    see Eq. 15 & 16 in Kipping 2013
    '''
    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2

def q_to_u(q1, q2):
    '''
    convert limb-darkening coefficients
    from q to u
    
    see Eq. 17 & 18 in Kipping 2013
    '''
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2*q2)
    return u1, u2


def transit_model_q(parameters, period, time, model=MandelAgol()):
    '''
    Compute flux using the Mandel-Agol model:
    
    \frac{I(\mu)}{I(1)} = 1 − u_1(1 − \mu) − u_2(1 − \mu)^2
    
    `parameters` must be a list containing:
    * k  : Rp/Rs: planet-star radius ratio
    * q1 : limb-darkening coefficient (for the linear term)
    * q2 : limb-darkening coefficient (for the quadratice term)
    * tc : transit center
    * a_s: a/Rs: scaled semi-major axis
    * b  : impact parameter
    '''
    k,q1,q2,tc,a,b = parameters
    
    #compute inclination
    inc   = np.arccos(b/a)
    #convert u to q
    u1,u2 = q_to_u(q1, q2)
    #evaluate the model
    m = model.evaluate(time, k, (u1,u2), tc, period, a, inc)
    
    return m

##############################################
#
# Empirical Limb-darkening (for prior)
#
##############################################

teff, uteff, logg, ulogg, feh, ufeh=5300,100, 4.460,0.06, 0.33,0.1

ldc_list     = []
ldc_err_list = []
ldp          = []

#for each band
for i in 'g*,r*,z*'.split(','):
    ldc = ld.claret(i, teff, uteff, logg, ulogg, feh, ufeh, n=int(1e4))
    ldp.append(ldc)
    
    #save in list
    #u1,u2
    ldc_list.append([ldc[0],ldc[2]]) #take first and third element of ldc 
    #uncertainties
    ldc_err_list.append([ldc[1],ldc[3]]) #take second and fourth element of ldc


##############################################
#
# Transit parameter optimization with 
# Maximum Likelihood Estimation (MLE)
#
##############################################

def obj(theta, p, t, f, err):
    '''
    objective function: chi-squared
    '''
    m = transit_model_q(theta, p, t)
    
    return np.sum(((f-m)/err)**2)

def rms(flux,flux_model):
    residual = flux-flux_model
    return np.sqrt(np.mean((residual)**2))


print('---computing MLE---')
fig,ax = pl.subplots(1,1,figsize=(15,8))

optimized_transit_params = {} 

n=0
for b,u,c in zip(bands,ldc_list,colors):
    print('{}-band'.format(b))
    df=data[b]
    
    flux = df[fluxcol]
    time = df.index
    err  = df[errcol]
    
    #plot raw data with vertical offset
    offset = n*0.03
    pl.errorbar(time, flux-offset, yerr=err, alpha=0.5) 
    
    #compute q from u found in limbdark
    q1_,q2_ = u_to_q(u[0],u[1])

    #compute flux before optimization
    transit_params     = [k_,q1_,q2_,tc_0,_a_s,_b]
    transit_model_before  = transit_model_q(transit_params, _P, time)
    #rms before
    rms_before = rms(flux,transit_model_before)
    print('rms before: {:.4f}'.format(rms_before))
    
    #optimize parameters
    result = op.minimize(obj, transit_params,
                         args=(_P, time, flux, err), method='nelder-mead')
    
    #compute flux after optimization
    transit_params_after     = np.copy(result.x)
    transit_model_after  = transit_model_q(transit_params_after, _P, time)
    #rms after
    rms_after = rms(flux,transit_model_after)
    print('rms after: {:.4f}\n'.format(rms_after))
    
    #plot transit models
    #before (faint red)
    ax.plot(time, transit_model_before-offset, 'r-', lw=3, alpha=0.5)
    #after (black)
    ax.plot(time, transit_model_after-offset, 'k-', lw=3, alpha=1)
    ax.legend(fontsize=fontsize)
    
    #dict of optimized transit parameters to be used later
    optimized_transit_params[b] = transit_params_after
    n+=1
    
ax.set_ylabel('Normalized Flux + Offset', fontsize=fontsize)
ax.set_xlabel('BJD(TDB)-2450000', fontsize=fontsize)
ax.legend(['before','after'], title='model optimization',fontsize=fontsize)
fig.savefig('optimized transit model.png')

##############################################
#
# Systematics model
#
##############################################

def binning(x,y,bins):
    t=np.linspace(x[0],x[-1], bins)
    y=np.interp(x=t, xp=x, fp=y)
    return t, y

def flux_ratios(x1, y1, x2, y2, bins):
    numerator   = binning(x1,y1,bins)[1] #get y-component only
    denominator = binning(x2,y2,bins)[1]
    return numerator/ denominator


#systematics model
def systematics_model(w, aux_vec, time):
    '''
    systematics model consists of linear combination
    of constant coefficients (computed here) 
    and auxiliary vectors:
    
    color1, color2, dx, err, vert_offset
    
    The functional form of the model is
    s = [np.sum(c[k] * x**k) for k in np.arange(N)]
    '''
    #make sure there are 4 aux. vectors
    assert len(aux_vec) == 4
    
    #unpack aux_vec
    color1, color2, dx, err = aux_vec
    #add vertical offset
    vert_offset = np.ones_like(dx)
    #construct X with time
    X = np.c_[color1, color2, dx, err, vert_offset, time]
    
    #compute systematics model
    sys_model = np.dot(X,w)
        
    return sys_model


def find_2_bands(b):
    '''
    this function return the 2 bands
    other than the given band
    
    e.g. if g, return r & z
    '''
    if b=='g':
        return ['r','z']
    elif b=='r':
        return ['g','z']
    elif b=='z':
        return ['g','r']
    else:
        sys.exit(1)


##############################################
#
# Computing systematics model coefficients
#
##############################################

fig,ax = pl.subplots(3,1,figsize=(18,15), sharex=True)

X_list = {}
w_list = {}
aux_vec_list = {}

n=0
for b,u,c in zip(bands,ldc_list,colors):
    print('{}-band'.format(b))
    df=data[b]
    
    flux = df[fluxcol]
    time = df.index
    err  = df[errcol]
    
    dx   = df['dx(pix)']
    dy   = df['dy(pix)']
    airmass = df['airmass']
    #fwhm    = df['fwhm(pix)']
    #sky     = df['sky(ADU)']
    #peak    = df['peak(ADU)']
    
    #plot flux raw data
    ax[n].errorbar(time, flux, yerr=err, 
                   label='data-sys model', alpha=0.5, color=c) 
    
    transit_params = optimized_transit_params[b]
    transit_model  = transit_model_q(transit_params, _P, time)
    
    #plot best fit transit model
    ax[n].plot(time, transit_model, 'k-', lw=3, label='transit model')
    data[b]['transit_model'] = transit_model 
    
    #plot residual with offset
    sys = flux / transit_model
    data[b]['flux/transit_model'] = sys
    
    ##ax[n].plot(time, resid+0.94, 'm-', lw=3, label='residual')
    ##ax[n].plot(time, sys+0.94, 'm-', lw=3, label='residual')
       
    #determine correct band
    b1,b2 = find_2_bands(b)
    
    #compute color
    color1 = flux_ratios(time,flux,data[b1].index,
                       data[b1][fluxcol], 
                       bins=len(df))
    color2 = flux_ratios(time,flux,data[b2].index,
                       data[b2][fluxcol], 
                       bins=len(df))
    #add color to original dataframe
    data[b]['color1'] = color1
    data[b]['color2'] = color2
    
    #add vertical offset
    vert_offset = np.ones_like(dx)
    
    #construct X with time
    X = np.c_[color1, color2, dx, err, vert_offset, time]
    X_list[b] = X
    
    #compute systematics model
    w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T, sys))
    w_list[b] = w
    
    #supply 4 vec, 2 others addded inside systematics_model function
    aux_vec = [color1, color2, dx, err]
    aux_vec_list[b] = aux_vec
    
    #evaluate model
    sys_model = systematics_model(w, aux_vec, time)
    
    data[b]['sys_model'] = sys_model
    
    resid = flux/sys_model - transit_model
    data[b]['residual'] = resid
    
    #compute rms
    rms_before = rms(flux, transit_model)
    #rms_after = rms(resid,sys_model)
    rms_after = rms(flux/sys_model,transit_model)
    
    print('rms (flux - transit_model): {:.4f}'.format(rms_before))
    print('rms (flux/sys model - transit_model): {:.4f}\n'.format(rms_after))
    print('rms difference: {:.4f}\n'.format(rms_before-rms_after))
    
    
    
    #plot systematics model
    ax[n].plot(time, sys_model, 'm-', lw=3, label='sys model')
    
    #plot corrected flux
    ax[n].errorbar(time, flux/sys_model-0.04, yerr=err, 
                   label='flux/sys model', alpha=0.5, color=c) 
    ax[n].plot(time, transit_model-0.04, 'k-', lw=3, label='transit model')
    
    ax[n].set_title('{}-band'.format(b), fontsize=fontsize)
    ax[n].legend(fontsize=12)
    ax[n].set_ylabel('Normalized Flux + Offset', fontsize=fontsize)
    n+=1
    
pl.xlabel('BJD(TDB)-2450000', fontsize=fontsize)
fig.savefig('systematics model.png')


##############################################
#
# Autocorrelation
#
##############################################
fig, ax = pl.subplots(nrows=3,ncols=1,figsize=(10,6), sharex=True)

n=0
for b in bands:
    df   = data[b]
    time = df.index
    
    #get residual computed earlier    
    resid_wo_sys = data[b]['residual']
        
    pd.plotting.autocorrelation_plot(resid_wo_sys, ax=ax[n])
    ax[n].set_title('Autocorrelation of residual with systematics model removed ({}-band)'.format(b))
    ax[n].set_ylim(-0.2,0.2)
    n+=1
fig.tight_layout()
fig.savefig('autocorrelation.png')



##############################################
#
# log Likelihood
#
##############################################


def loglike(params_full, p, t, f, err, aux_vec, 
            ret_mod=False, ret_sys=False, ret_full = False):
    '''
    * computes the log likelihood given the optimized transit and model parameters
    * either or both transit and systematics models can also be returned
    '''
    
    m = transit_model_q(params_full[:6], p, t)
    
    #color1,color2,dx,err = aux_vec
    s = systematics_model(params_full[6:], aux_vec, t) # #add sys model
    
    if ret_mod:
        return m
    if ret_sys:
        return s
    if ret_full:
        return m*s
    
    resid = f / s - m
    
    #-0.5*(np.sum((resid)**2 * np.exp(-2*ls) + 2*ls))
    C = np.log(2*np.pi)
    
    return -0.5*(np.sum(np.log(err) + C + (resid/err)**2))

#negative log-likelihood
nll = lambda *x: -loglike(*x)


##############################################
#
# Simultaneous modeling optimization (MLE)
#
##############################################


#parameters vector: 6 transit, 4+2 systematics

fig = pl.figure(figsize=(15,9))

save_model = True

n=0
for b in bands:
    df=data[b]

    flux = df[fluxcol]
    time = df.index

    #sys mod params
    airmass   = df['airmass']
    err       = df['err(r=11.0)']
    dx        = df['dx(pix)']
    #dy        = df['dy(pix)']
    print('--{}--'.format(b))
    
    #transit params computed before
    transit_params = optimized_transit_params[b]
    
    #weights computed before
    w = w_list[b]
    
    #aux_vec saved before
    aux_vec = aux_vec_list[b]
    
    #combine optimized transit params and sys params
    full_params = np.concatenate((transit_params, w), axis=0)
    
    #compute nll
    print ("NLL before: {}".format(nll(full_params, 
                                       _P, 
                                       time, 
                                       flux, 
                                       err, 
                                       aux_vec)))
    #MLE optimization of transit+sys parameters
    result = op.minimize(nll, full_params,    
                         args=(_P, time, flux, err, aux_vec),
                         method='nelder-mead')
    
    #result of optimization
    print ("NLL after: {}".format(nll(result.x,   
                                      _P, 
                                      time, 
                                      flux, 
                                      err, 
                                      aux_vec)))
    
    #compute models
    full_model    = loglike(result.x, _P, time, flux, err, aux_vec, 
                            ret_full=True)
    transit_model = loglike(result.x, _P, time, flux, err, aux_vec, 
                     ret_mod=True)
    sys_model     = loglike(result.x, _P, time, flux, err, aux_vec, 
                     ret_sys=True)
    
    resid = flux/sys_model - transit_model
    
    if save_model:
        data[b]['transit_model'] = transit_model
        data[b]['sys_model']     = sys_model
        data[b]['full_model']    = full_model
        data[b]['residual']      = resid
    
    rms   = np.sqrt(np.mean(resid**2))
    
    ax = pl.subplot(3,1,n+1)
    #plot corrected data
    ax.plot(time, flux, 'ko', alpha=0.5, label='raw data')
    ax.set_title('raw data & transit+systematics models ({}-band)'.format(b), fontsize=fontsize)
    #plot transit+sys models
    ax.plot(time, full_model, 'r-', lw=2, label='transit & sys model');
    n+=1
    
pl.legend(fontsize=fontsize)
fig.tight_layout()
fig.savefig('full model (MLE).png')


##############################################
#
# Simultaneous modeling optimization (MLE)
#
##############################################


def tshape_approx(a, k, b):
    """
    Seager & Mallen-Ornelas 2003, eq. 15
    """
    i = np.arccos(b/a)
    alpha = (1 - k)**2 - b**2
    beta = (1 + k)**2 - b**2
    return np.sqrt( alpha / beta )


def max_k(tshape):
    """
    Seager & Mallen-Ornelas 2003, eq. 21
    """
    return (1 - tshape) / (1 + tshape)


def rhostar(p, a):
    """
    Eq.4 of Kipping 2014. 
    Assumes circular orbit & Mp<<Ms
    
    http://arxiv.org/pdf/1311.1170v3.pdf
    
    c.f. Seager & __ 2002
    """
    p = p * units.d
    gpcc = units.g / units.cm ** 3
    rho_mks = 3 * np.pi / constants.G / p ** 2 * a ** 3
    return rho_mks.to(gpcc).value



##############################################
#
# log prior
#
##############################################

def logprior(full_params,u_prior=None):
    '''
    full_paras: transit+systematic model parameters
    up: limb-darkening prior for u1,u2
    '''
    #unpack transit parameters
    k,q1,q2,tc,a_s,impact_param = full_params[:6]
    inc=np.arccos(impact_param/a_s)
    
    tshape = tshape_approx(a_s, k_, _b)
    kmax = max_k(tshape)
    
    rho_s = rhostar(_P, a_s)

    #Uniform priors: return log 0= inf if outside interval
    if  q1  < 0 or q1 > 1 or \
        q2  < 0 or q2 > 1 or \
        k   < 0 or k  > kmax or \
        impact_param   < 0 or impact_param  > 1 or \
        inc > np.pi/2     or \
        a_s < 0 or a_s  > 13 or \
        rho_s < 1 or rho_s > 10 or \
        tc < tc_0-_t14/2 or tc > tc_0+_t14:
        
        #print('off limits encountered in logprior')
        return -np.inf
    
    #if they are inside interval, add log 1 = 0 to logprior
    
    #logprior
    lp = 0
    
    #Normal priors on q1,q2 with sigma=uncertainty from Claret+2012
    u1,u2 = q_to_u(q1,q2)
    
    if u_prior is not None:
        #evaluate log of N(u,sigma)
        sigma1=u_prior[1]
        sigma2=u_prior[3]
        #add to logprior
        lp += np.log(stats.norm.pdf(u1, loc=u_prior[0], scale=sigma1))
        lp += np.log(stats.norm.pdf(u2, loc=u_prior[2], scale=sigma2))
    
    #what are our priors for systematics model parameters?
    
    return lp


##############################################
#
# log probability = log prior + log likelihood
#
##############################################


def logprob(full_params,
            time_list,
            flux_list,
            period,
            color1_list,
            color2_list,
            dx_list,
            err_list,
            ldc_prior=None):
    
    t1,t2,t3                   = time_list
    f1,f2,f3                   = flux_list
    color1g,color1r,color1z    = color1_list
    color2g,color2r,color2z    = color2_list
    dx1,dx2,dx3                = dx_list
    err1,err2,err3             = err_list
    #airmass1,airmass2,airmass3 = airmass_list
    #ycen1,ycen2,ycen3         = dy_list
    
    #unpack full params for 3 bands
    k_g,k_r,k_z,tc,a,impact_param,q1g,q2g,q1r,q2r,q1z,q2z,\
    w0g,w1g,w2g,w3g,w4g,w5g,\
    w0r,w1r,w2r,w3r,w4r,w5r,\
    w0z,w1z,w2z,w3z,w4z,w5z = full_params
    
    #set up params list for each band
    theta1 = [k_g,q1g,q2g,tc,a,impact_param,w0g,w1g,w2g,w3g,w4g,w5g]
    theta2 = [k_r,q1r,q2r,tc,a,impact_param,w0r,w1r,w2r,w3r,w4r,w5r]
    theta3 = [k_z,q1z,q2z,tc,a,impact_param,w0z,w1z,w2z,w3z,w4z,w5z]
    
    #set up auxiliary vector for each band
    aux_vec1 = color1g, color2g, dx1, err1
    aux_vec2 = color1r, color2r, dx2, err2
    aux_vec3 = color1z, color2z, dx3, err3
    
    #sum loglike for each band
    ll  = loglike(theta1, period, t1, f1, err1, aux_vec1)
    ll += loglike(theta2, period, t2, f2, err2, aux_vec2)
    ll += loglike(theta3, period, t3, f3, err3, aux_vec3)
    
    if ldc_prior is not None:
        lp  = logprior(theta1, u_prior=ldc_prior[0])
        lp += logprior(theta2, u_prior=ldc_prior[1])
        lp += logprior(theta3, u_prior=ldc_prior[2])
    else:
        #no ldc prior (if stellar parameters not known)
        lp  = logprior(theta1)
        lp += logprior(theta2)
        lp += logprior(theta3)
    
    if np.isnan(ll).any():
        #print('NaN encountered in loglike')
        return -np.inf
    
    #total: sum of prior and likelihood
    return lp + ll

#negative log prob
nlp = lambda *x: -logprob(*x)


#ignore tc, a_s, b
k_g,q1g,q2g,_,_,_ = optimized_transit_params['g']
k_r,q1r,q2r,_,_,_ = optimized_transit_params['r']
k_z,q1z,q2z,_,_,_ = optimized_transit_params['z']

times    = []
fluxes   = []
colors1  = []
colors2  = []
errs     = []
dxs      = []
#dys      =[]
#airmasses= []

for b in bands:
    #sys mod params
    df = data[b]
    times.append(df.index)
    fluxes.append(df['flux(r=11.0)'])
    colors1.append(df['color1'])
    colors2.append(df['color2'])
    errs.append(df['err(r=11.0)'])    
    dxs.append(df['dx(pix)'])    

color1_coeffs= []
color2_coeffs= []
dx_coeffs    = []
err_coeffs   = []
vert_offsets = []
time_coeffs  = []

for b in bands:
    color1_coeffs.append(w_list[b][0])
    color2_coeffs.append(w_list[b][1])
    dx_coeffs.append(w_list[b][2])
    err_coeffs.append(w_list[b][3])
    vert_offsets.append(w_list[b][4])
    time_coeffs.append(w_list[b][5])

#unpack
w0g,w0r,w0z = color1_coeffs
w1g,w1r,w1z = color2_coeffs
w2g,w2r,w2z = dx_coeffs
w3g,w3r,w3z = err_coeffs
w4g,w4r,w4z = vert_offsets
w5g,w5r,w5z = time_coeffs

full_params = [k_g,k_r,k_z,tc_0,_a_s,_b,q1g,q2g,q1r,q2r,q1z,q2z,\
                w0g,w1g,w2g,w3g,w4g,w5g,\
                w0r,w1r,w2r,w3r,w4r,w5r,\
                w0z,w1z,w2z,w3z,w4z,w5z]


print('log probability:')
logprob(full_params,
        times,
        fluxes,
        _P,
        colors1,
        colors2,
        dxs,
        errs,
        ldc_prior=ldp)


##############################################
#
# reduced chi-square and beta factor
#
##############################################


def chisq(resid, sig, ndata=None, nparams=None, reduced=False):
    if reduced:
        assert ndata is not None and nparams is not None
        dof = ndata - nparams
        return np.sqrt(sum((resid / sig)**2)/ (dof))
    else:
        return sum((resid / sig)**2)

reduced_chi2 = {}

newerrcols = []

for b in bands:
    print('{}-band'.format(b))
    df = data[b]
    
    ndata = len(df)
    nparams = len(full_params)
    
    resid = df['residual']
    err   = df['err(r=11.0)']
    
    chi2 = chisq(resid, err, ndata, nparams)
    red_chi2 = chisq(resid, err, ndata, nparams, reduced=True)
    
    #save
    reduced_chi2[b] = red_chi2
    col='err(r=11.0)*{:.2f}'.format(red_chi2)
    newerrcols.append(col)
    data[b][col] = err*red_chi2
    
    print('chi2 ={:.2f}'.format(chi2))
    print('reduced chi2 ={:.4f}'.format(red_chi2))


def binned(a, binsize, fun=np.mean):
    a_b = []
    for i in range(0, a.shape[0], binsize):
        a_b.append(fun(a[i:i+binsize], axis=0))
        
    return a_b

def beta(residuals, timestep, start_min=5, stop_min=20, return_dict=False):
    """
    residuals : flux/sys_model - transit_model
    timestep  : time interval between datapoints in seconds
    
    Final beta is computed by taking the median of the betas:
                np.nanmedian(betas)
    """

    assert timestep < start_min * 60
    ndata = len(residuals)
    
    sigma1 = np.std(residuals)

    min_bs = int(start_min * 60 / timestep)
    max_bs = int(stop_min  * 60 / timestep)

    betas      = []
    betas_dict = {}
    for bs in np.arange(min_bs, max_bs + 1):
        nbins = ndata / bs
        sigmaN_theory = sigma1 / np.sqrt(bs) * np.sqrt( nbins / (nbins - 1) )
        
        #binning
        sigmaN_actual = np.std(binned(residuals,bs))
        beta = sigmaN_actual / sigmaN_theory
        
        betas_dict[bs] = beta
        betas.append(beta)
        
    if return_dict:
        return betas_dict
    else:
        return np.nanmedian(betas)


beta_factor = {}

exptimes = [60,30,60] #sec

for b,col,ts in zip(bands,newerrcols,exptimes):
    df = data[b]
    resid = df['residual'].values
    
    beta_fac = beta(resid, ts, start_min=5, stop_min=20)
    beta_factor[b]= beta_fac
    
    err = df[col]
    
    #inflate error once more (after rescaling red. chi-squared)
    inflated_err = err * beta_fac
    
    #add to df 
    data[b]['err_inflated'] = inflated_err
    
    print('beta factor ={:.4f}'.format(beta_fac))


fluxcol =  'flux(r=11.0)'
errcol  =  'err_inflated' #'err(r=11.0)'

colors='b,g,r'.split(',')
fig,ax = pl.subplots(1,1,figsize=(15,4))

n=0
for b,c in zip(bands,colors):
    df = data[b]
    offset = n*0.02
    
    time = df.index
    flux = df[fluxcol]
    err  = df[errcol]
    
    ax.errorbar(time, flux-offset, yerr=err, label=b, color=c, alpha=0.5)
    ax.set_ylabel('Normalized flux', fontsize=fontsize)
    n+=1
pl.legend(loc='lower left', fontsize=fontsize)
pl.xlabel('BJD(TDB)-2450000', fontsize=fontsize)
fig.savefig('inflated errors.png')

#update error
errs=[]
for b in bands:
    df=data[b]
    errs.append(df['err_inflated'])  



##############################################
#
# MCMC with emcee
#
##############################################

print('---setting up MCMC---')
ndim = len(full_params)
nwalkers = 8 * ndim if ndim > 12 else 16 * ndim

#input to the model
args = [times, fluxes, _P, colors1, colors2, dxs, errs, ldp]

sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=args, threads=1)

pos0 = []
#random numbers "around" the initial values
for i in range(nwalkers):
    #pick a random number
    rnum = 1e-8 * np.random.randn(ndim)
    #add to each initial value of parameter
    new_param_vector = np.array(full_params) +rnum
    #append
    pos0.append(new_param_vector)



from tqdm import tqdm

nsteps1 = 1000

#begin mcmc: 1st stage
for pos,lnp,rstate in tqdm(sampler.sample(pos0, iterations=nsteps1)):
    pass



#load result of previous mcmc runs
#theta_sys0 = np.load('theta_post.csv')
#pos0 = [np.array(theta_sys0) + 1e-8 * np.random.randn(ndim) for i in range(nwalkers)]


#visualize 1st stage
param_names='k_g,k_r,k_z,tc,a,b,q1g,q2g,q1r,q2r,q1z,q2z,w0g,w1g,w2g,w3g,w4g,w5g,w0r,w1r,w2r,w3r,w4r,w5r,w0z,w1z,w2z,w3z,w4z,w5z'.split(',')

chain=sampler.chain
nwalkers, nsteps, ndim = chain.shape
fig, axs = pl.subplots(ndim, 1, figsize=(15,ndim/1.5), sharex=True)
#ls, lc = ['-','--','--'], ['k', '0.5', '0.5']
#percs = [np.percentile(sampler.chain2[:,:,i], [50,16,84], 0) for i in range(ndim)]
[axs.flat[i].plot(c, drawstyle='steps', color='k', alpha=4./nwalkers) for i,c in enumerate(chain.T)]
#[[axs.flat[i].plot(percs[i][j], c=lc[j], ls=ls[j]) for j in range(3)] for i in range(ndim)]
[axs.flat[i].set_ylabel(l) for i,l in enumerate(param_names)]  
pl.xlabel('nsteps')
fig.savefig('chain1.png')

#send chain.png from server to local
#os.system("scp Jerome@esptodai.astron.s.u-tokyo.ac.jp:/home/Jerome/github/muscat/HAT-P-12b/*.png .")

#check and discard burn-in stage
#import pdb
#pdb.set_trace()
# set burn: burn = 1000, say then press c

#burn = 5000
#chain=sampler.chain[:,burn:,:]

#save chain
loc='.'
with gzip.GzipFile(os.path.join(loc,'chain1.npy.gz'), "w") as g1:
    np.save(g1, chain)

with gzip.GzipFile(os.path.join(loc,'lnp1.npy.gz'), "w") as g2:
    np.save(g2, sampler.flatlnprobability)
#np.allclose(sample_chain.shape,chain.shape)


#restart mcmc: 1st stage
sampler.reset()
nsteps2= 10000

for pos2,lnp2,rstate in tqdm(sampler.sample(pos, iterations=nsteps2)):
    pass

#save chain
chain=sampler.chain
loc='.'
with gzip.GzipFile(os.path.join(loc,'chain2.npy.gz'), "w") as g1:
    np.save(g1, chain)

with gzip.GzipFile(os.path.join(loc,'lnp2.npy.gz'), "w") as g2:
    np.save(g2, sampler.flatlnprobability)
#np.allclose(sample_chain.shape,chain.shape)

#visualize 2nd stage
nwalkers, nsteps, ndim = chain.shape
fig, axs = pl.subplots(ndim, 1, figsize=(15,ndim/1.5), sharex=True)
#ls, lc = ['-','--','--'], ['k', '0.5', '0.5']
#percs = [np.percentile(sampler.chain2[:,:,i], [50,16,84], 0) for i in range(ndim)]
[axs.flat[i].plot(c, drawstyle='steps', color='k', alpha=4./nwalkers) for i,c in enumerate(chain.T)]
#[[axs.flat[i].plot(percs[i][j], c=lc[j], ls=ls[j]) for j in range(3)] for i in range(ndim)]
[axs.flat[i].set_ylabel(l) for i,l in enumerate(param_names)]  
pl.xlabel('nsteps')
fig.savefig('chain2.png')
