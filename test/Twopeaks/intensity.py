import jax.numpy as jnp
import numpy as np
import numpyro 
import numpyro.distributions as dist
import jax
import jax.scipy.stats as jsst

def pop_model(mobs, sigma):
    
    Nobs = len(mobs)
    
    mu2 = numpyro.sample('mu2', dist.Uniform(low=50.0, high=70.0))
    sigma2 = numpyro.sample('sigma2', dist.Uniform(low=0.1, high=10.0))
    
    sigma22 = jnp.sqrt(sigma**2+sigma2**2)
    var = jsst.norm.logpdf(mobs, mu2, sigma22)
    c = -jnp.log1p(-jsst.norm.cdf(50.0, mu2, sigma22))
    _ = numpyro.factor('pos', Nobs*c + jnp.sum(var))