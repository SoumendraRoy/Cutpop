import jax.numpy as jnp
import numpy as np
import numpyro 
import numpyro.distributions as dist
import jax
import jax.scipy.stats as jsst
import jax.scipy.special as jss

def pop_model_true(mtrue, ztrue):
    
    # m hyperparameters
    f1 = numpyro.sample('f1', dist.Uniform(low=0.01, high=0.99))
    mu1 = numpyro.sample('mu1', dist.Uniform(low=30.0, high=40.0))
    sigma1 = numpyro.sample('sigma1', dist.Uniform(low=0.1, high=6.0))
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=3.0))
    
    # z hyperparameters
    a = numpyro.sample('a', dist.Uniform(low=1.0, high=10.0))
    b = numpyro.sample('b', dist.Uniform(low=1.0, high=20.0))
    
    mpdf = f1*jsst.norm.pdf(mtrue, mu1, sigma1) + (1-f1)*jsst.norm.pdf(mtrue, mu2, sigma2)
    zlogpdf = jsst.beta.logpdf(ztrue/10.0, a, b)
    _ = numpyro.factor('pos', jnp.sum(jnp.log(mpdf) + zlogpdf))
    
def pop_model_mcmc_samples(ms, zs):
    # m hyperparameters
    f1 = numpyro.sample('f1', dist.Uniform(low=0.01, high=0.99))
    mu1 = numpyro.sample('mu1', dist.Uniform(low=30.0, high=40.0))
    sigma1 = numpyro.sample('sigma1', dist.Uniform(low=0.1, high=6.0))
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=3.0))
    
    # z hyperparameters
    a = numpyro.sample('a', dist.Uniform(low=1.0, high=10.0))
    b = numpyro.sample('b', dist.Uniform(low=1.0, high=20.0))
    
    mpdf = f1*jsst.norm.pdf(ms, mu1, sigma1) + (1-f1)*jsst.norm.pdf(ms, mu2, sigma2)
    zpdf = jsst.beta.pdf(zs/10.0, a, b)
    var = jnp.sum(mpdf*zpdf, axis=1)
    _ = numpyro.factor('pos', jnp.sum(jnp.log(var)))
    
def pop_model_mcmc_samples_w_sel(ms_det, zs_det, mdet, zdet):
    # m hyperparameters
    f1 = numpyro.sample('f1', dist.Uniform(low=0.01, high=0.99))
    mu1 = numpyro.sample('mu1', dist.Uniform(low=30.0, high=40.0))
    sigma1 = numpyro.sample('sigma1', dist.Uniform(low=0.1, high=6.0))
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=3.0))
    
    # z hyperparameters
    a = numpyro.sample('a', dist.Uniform(low=1.0, high=10.0))
    b = numpyro.sample('b', dist.Uniform(low=1.0, high=25.0))
    
    mlogpdf = jnp.log(f1*jsst.norm.pdf(ms_det, mu1, sigma1) + (1-f1)*jsst.norm.pdf(ms_det, mu2, sigma2))
    zlogpdf = jsst.beta.logpdf(zs_det/10.0, a, b)
    var = jss.logsumexp(mlogpdf + zlogpdf, axis=1)
    
    # selection effect
    mlogpdf_det = jss.logsumexp(jnp.array([jnp.log(f1)+jsst.norm.logpdf(mdet, mu1, sigma1),jnp.log(1-f1)+jsst.norm.logpdf(mdet, mu2, sigma2)]), axis=0)
    zlogpdf_det = jsst.beta.logpdf(zdet/10.0, a, b)
    var_det = jss.logsumexp(mlogpdf_det + zlogpdf_det)
    
    _ = numpyro.factor('pos', jnp.sum(var)-len(ms_det)*var_det)
    
def pop_model_higher_mass_peak(ms_det, zs_det, mdet, zdet):
    # m hyperparameters
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=3.0))
    
    # z hyperparameters
    a = numpyro.sample('a', dist.Uniform(low=1.0, high=10.0))
    b = numpyro.sample('b', dist.Uniform(low=1.0, high=25.0))
    
    mlogpdf = jsst.norm.logpdf(ms_det, mu2, sigma2)
    zlogpdf = jsst.beta.logpdf(zs_det/10.0, a, b)
    var = jss.logsumexp(mlogpdf + zlogpdf, axis=1)
    
    # selection effect
    mlogpdf_det = jsst.norm.logpdf(mdet, mu2, sigma2)
    zlogpdf_det = jsst.beta.logpdf(zdet/10.0, a, b)
    var_det = jss.logsumexp(mlogpdf_det + zlogpdf_det)
    
    _ = numpyro.factor('pos', jnp.sum(var)-len(ms_det)*var_det)