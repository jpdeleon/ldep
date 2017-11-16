'''
This is a compilation of functions used for analyzing MuSCAT
data.  
'''

#######################################
#
#   Light curve modeling
#
#######################################

from pytransit import MandelAgol

def transit_model_q(parameters, period, time, model=MandelAgol()):
    '''
    Compute flux using the Mandel-Agol model analytic light curve model:
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

def loglike(params, p, t, f, err, nparams=6):
    '''
    log likelihood of the transit model
    '''
    assert len(params) == nparams
    m = transit_model_q(params, p, t)
        
    resid = f - m
    #-0.5*(np.sum((resid)**2 * np.exp(-2*ls) + 2*ls))
    C = np.log(2*np.pi)
    
    return -0.5*(np.sum(np.log(err) + C + (resid/err)**2))


def logprob(full_params,
            time,
            flux,
            period,
            color1,
            color2,
            dx,
            err,
            ldc_prior=None,
            qerr=False):
    '''
    log probability = logprior + loglikelihood
    '''
    #unpack full params for 3 bands
    
    if qerr:
        k,tc,a_s,impact_param,lsjit,q1,q2,w0,w1,w2,w3,w4,w5 = full_params
    else:
        k,tc,a_s,impact_param,q1,q2,w0,w1,w2,w3,w4,w5 = full_params
    
    #set up auxiliary vector for each band
    aux_vec = color1, color2, dx, err
    
    #sum loglike for each band
    ll  = loglike(full_params, period, time, flux, err, aux_vec)
    
    if ldc_prior is not None:
        lp  = logprior(full_params, u_prior=ldc_prior)
    else:
        #no ldc prior (if stellar parameters not known)
        lp  = logprior(full_params)
    
    if np.isnan(ll).any():
        #print('NaN encountered in loglike')
        return -np.inf
    
    #total: sum of prior and likelihood
    return lp + ll

#negative log prob
nlp = lambda *x: -logprob(*x)


#######################################
#
#   Optimization
#
#######################################

import scipy.optimize as op

def obj(theta, p, t, f, err, nparams=6):
    '''
    objective function: chi-square
    '''
    assert len(theta) == nparams
    
    m = transit_model_q(theta[:6], p, t)
    
    return np.sum(((f / m)/err)**2)

import numpy as np

def simple_ols(x, y, intercept=True):
    """
    Simple OLS with no y uncertainties.
    x : array-like, abscissa
    y : array-like, ordinate
    """
    if intercept:
        X = np.c_[np.ones_like(x), x]
    else:
        X = x
        
    #np.dot( np.dot( np.linalg.inv( np.dot(X.T, X) ), X.T), y )
    return np.linalg.solve(np.dot(X.T,X), np.dot(X.T, y))


#######################################
#
#   Equations related to light curve 
#
#######################################


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


#######################################
#
#   Statistics
#
#######################################

def rms(flux, flux_model):
    '''
    computes the root-mean-square error
    '''
    residual = flux-flux_model
    #return np.sqrt((residual**2).sum()/residual.size)
    
    #(flux/sys_model - transit_model) / error,
    #or (flux - transit_model*sys_model) / error
    return np.sqrt(np.mean((residual)**2))

def chisq(resid, sig, ndata=None, nparams=None, reduced=False):
    '''
    computes (reduced) chi-square
    '''
    if reduced:
        assert ndata is not None and nparams is not None
        dof = ndata - nparams
        return sum((resid / sig)**2) / (dof)
    else:
        return sum((resid / sig)**2)

def bic(lnlike, ndata, nparam):
    '''
    Bayesian Information Criterion for model selection
    '''
    return -2 * lnlike + nparam * np.log(ndata)

def gelman_rubin(chains, verbose=False):
    '''
    computes the Gelman-Rubin statistic for MCMC chain convergence, s
    Ideally, s < 1.05
    '''
    assert chains.ndim == 3
    nn = chains.shape[1]
    mean_j = chains.mean(axis=1)
    var_j = chains.var(axis=1)
    B = nn * mean_j.var(axis=0)
    W = var_j.mean(axis=0)
    R2 = ( W*(nn-1)/nn + B/nn ) / W
    return np.sqrt(R2)



#######################################
#
#   Miscellaneous
#
#######################################

def binning(x,y,bins):
    '''
    bin (x,y) by interpolation
    '''
    t=np.linspace(x[0],x[-1], bins)
    y=np.interp(x=t, xp=x, fp=y)
    return t, y

def flux_ratios(x1, y1, x2, y2, bins):
    numerator   = binning(x1,y1,bins)[1] #get y-component only
    denominator = binning(x2,y2,bins)[1]
    return numerator/ denominator

def find_2_bands(b):
    '''
    this function returns the 2 bands
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
