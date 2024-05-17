import jax.numpy as jnp
import numpy as np
import numpyro 
import numpyro.distributions as dist
import jax
import jax.scipy.stats as jsst

def powerlaw_pdf(m, alpha=2.35, a=5.0, b=150.0):
    norm = (a - (a/b)**alpha*b)/(a*(alpha-1))
    return (a/m)**alpha/a/norm

def integrand(mobs, m, sigma=1.0, alpha=2.35, a=5.0, b=150.0):
    return jsst.norm.pdf(m, mobs, sigma)*powerlaw_pdf(m)

def pop_model_full_model_with_errorbar(mobs, sigma):
    f1 = 0.2
    f2 = 0.1
    #f1 = numpyro.sample('f1', dist.Uniform(low=0.01, high=0.5))
    mu1 = numpyro.sample('mu1', dist.Uniform(low=30.0, high=40.0))
    sigma1 = numpyro.sample('sigma1', dist.Uniform(low=1.0, high=6.0))
    #f2 = numpyro.sample('f2', dist.Uniform(low=0.01, high=0.5))
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=10.0))
    
    sigma11 = jnp.sqrt(sigma**2+sigma1**2)
    sigma22 = jnp.sqrt(sigma**2+sigma2**2)
    mdummy = jnp.linspace(mobs-4.0*sigma, mobs+4.0*sigma, 10000)
    integrate = jnp.trapz(integrand(mobs[None, :], mdummy), mdummy, axis=0)
    var = f1*jsst.norm.pdf(mobs, mu1, sigma11) + f2*jsst.norm.pdf(mobs, mu2, sigma22) + (1-f1-f2)*integrate
    _ = numpyro.factor('pos', jnp.sum(jnp.log(var)))
    
def pop_model_higher_mass_with_errorbar(mobs, sigma):
    f1 = 0.2
    f2 = 0.1
    #f2 = numpyro.sample('f2', dist.Uniform(low=0.01, high=0.5))
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=60.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=10.0))
    
    sigma22 = jnp.sqrt(sigma**2+sigma2**2)
    mdummy = jnp.linspace(mobs-4.0*sigma, mobs+4.0*sigma, 10000)
    integrate = jnp.trapz(integrand(mobs[None, :], mdummy), mdummy, axis=0)
    var = f2*jsst.norm.pdf(mobs, mu2, sigma22) + (1-f1-f2)*integrate
    _ = numpyro.factor('pos', jnp.sum(jnp.log(var)))