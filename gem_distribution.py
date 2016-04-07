import numpy

alpha = 1

betas = []

for k in range(100):
    beta_k = numpy.random.beta(1, alpha)
    betas.append(beta_k)

print betas

one_minus_betas = numpy.ones(100) - betas

pis = []

for k,beta_k in enumerate(betas):
    print k,beta_k
    if k == 0:
        pis.append(beta_k)
    elif k >= 1:
        pi_k = beta_k * numpy.cumprod(one_minus_betas[0:k-1])[-1] #inefficient, but whatever
        pis.append(pi_k)
    else:
        print 'something weird is happening; shouldnt get here'
print pis
