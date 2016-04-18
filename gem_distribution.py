# The following code uses variable names from Section 4.1 of 
# Hierarchical Dirichlet Processes by Yee Whye Teh, Jordan, Beal, & Blei
# http://www.cs.berkeley.edu/~jordan/papers/hdp.pdf



#GENERATE TRUNCATED GEM DISTRIBUTION

import numpy


def gem_distribution(alpha,number_of_clusters):
    #alpha = 1
    ##number_of_sticks = 10

    #since this is a stick-breaking construction of GEM
    number_of_sticks = number_of_clusters 

    betas = []

    for k in range(number_of_sticks):
        beta_k = numpy.random.beta(1, alpha)
        betas.append(beta_k)

    print betas

    one_minus_betas = numpy.ones(number_of_sticks) - betas

    pis = []

    for k,beta_k in enumerate(betas):
        print k,beta_k
        if k == 0:
            pis.append(beta_k)
        elif k >= 1:
            #print one_minus_betas[0:k-1]
            pi_k = beta_k * numpy.cumprod(one_minus_betas[0:k])[-1] #inefficient, but whatever
            pis.append(pi_k)
        else:
            print 'something weird is happening; shouldnt get here'
    return pis

#test call
gem_distribution(1,10)