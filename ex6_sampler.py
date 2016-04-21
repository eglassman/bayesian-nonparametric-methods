# Python version of ex6_sampler.R
# Elena Glassman, elena.glassman@gmail.com

import numpy as np #nice for matrices

# generate Gaussian mixture model data
# Args:
#  Ndata: number of data points to generate
#  sd: covariance matrix of data points around the
#      cluster-specific mean is [sd^2, 0; 0, sd^2];
#      i.e. this is the standard deviation in either direction
#
# Returns:
#  x: an Ndata x 2 matrix of data points
#  z: an Ndata-long vector of cluster assignments
#  mu: a K x 2 matrix of cluster means,
#      where K is the number of clusters
def generate_gaussian_mix(Ndata,sd):

    # matrix of cluster centers: one in each quadrant
    mu = np.matrix('3 3; 3 -3; -3 3; -3 -3')

    # vector of component frequencies
    rho = np.array([0.4, 0.3, 0.2, 0.1])

    # assign each data point to a component
    z = np.random.choice(range(len(rho)), size=Ndata, replace=True, p=rho)

    # draw each data point according to the cluster-specific
    # likelihood of its component
    #x = np.random.normal(mu[], sigma, 1000)
    mu_dim0s = np.squeeze(np.asarray(mu[z,0]))
    mu_dim1s = np.squeeze(np.asarray(mu[z,1]))

    x_dim0s = np.random.normal(mu_dim0s,sd,Ndata)
    x_dim1s = np.random.normal(mu_dim1s,sd,Ndata)

    x = np.c_[x_dim0s,x_dim1s]

    data_dict = {}
    data_dict['x']=x
    data_dict['z']=z
    data_dict['mu']=mu
    data_dict['rho']=rho
    
    return data_dict

#create some fake data
gaussian_mix_dict = generate_gaussian_mix(1000,1)


#plot this data
import numpy as np
import matplotlib.pyplot as plt

#maps z's to values between zero and 1, not including zero or 1
x0s = gaussian_mix_dict['x'][:,0]
x1s = gaussian_mix_dict['x'][:,1]
z = gaussian_mix_dict['z']
rho = gaussian_mix_dict['rho']
colors = z*1.0/rho.size+1.0/(rho.size*2)

plt.scatter(x0s, x1s, c=colors)
plt.show()

##########################
#Gibbs Sampler!

#Adapted from https://github.com/danstowell/gmphd/blob/master/gmphd.py
def dmvnorm(x,loc,cov):
    #"Evaluate a multivariate normal, given a location (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
    #loc = np.array(loc, dtype=myfloat)
    #cov = np.array(cov, dtype=myfloat)
    #x = array(x, dtype=myfloat)
    k = len(loc)
    pi = np.pi
    part1 = (2.0 * pi) ** (-k * 0.5)
    part2 = np.power(np.linalg.det(cov), -0.5)
    dev = x - loc
    part3 = np.exp(-0.5 * np.dot(np.dot(np.transpose(dev), np.linalg.inv(cov)), dev))
    return part1 * part2 * part3

def run_gibbs(data,sd,sd_prior,initz,alpha):
    # don't exceed this many Gibbs iterations
    maxIters = 5

    # the algorithm will pause and plot after this
    # iteration number; 0 ensures it will plot right off
    minPauseIter = 0

    # dimension of the data points (number of columns)
    data_dim = data.shape[1]
    # cluster-specific covariance matrix
    Sig = np.diag(sd**2*np.ones(data_dim))
    # prior covariance matrix
    Sig0 = np.diag(sd_prior**2*np.ones(data_dim))
    # cluster-specific precision (Sig^{-1})
    Prec = np.linalg.inv(Sig)
    # prior precision (Sig^{-1})
    Prec0 = np.linalg.inv(Sig0)
    # prior mean on cluster parameters
    mu0 = np.zeros(2)
    # number of data points
    number_of_datapoints = data.shape[0]
    
    print 'initialization'
    print alpha
    print Sig
    print Sig0
    print Prec
    print Prec0
    print mu0

    # initialize the sampler
    z = initz  # initial cluster assignments
    cluster_counts = []
    for cluster_label in np.unique(z):
        vector_of_ones = [1 for z_elem in z if z_elem==cluster_label]
        cluster_counts.append(sum(vector_of_ones))
    number_of_clusters = len(cluster_counts)   # initial number of clusters
    
    # run the Gibbs sampler
    for iteration in range(maxIters):
        
        print
        print 'iteration:',iteration
        print
        
        # take a Gibbs step at each data point
        for n in range(number_of_datapoints):
            
            print
            print 'datapoint:', n
            
            # get rid of the nth data point
            print 'cluster counts',cluster_counts
            cluster_id = z[n]
            cluster_counts[cluster_id] = cluster_counts[cluster_id] - 1

            # if the nth data point was the only point in a cluster,
            # get rid of that cluster
            if cluster_counts[cluster_id]==0:
                last_cluster = number_of_clusters - 1
                #re-assign last cluster to empty cluster's id
                z[z==last_cluster] = cluster_id
                #update the cluster_counts array
                cluster_counts[cluster_id] = cluster_counts[last_cluster]
                #remove original entry for last_cluster in cluster_counts
                cluster_counts = cluster_counts[:-1]
                number_of_clusters = number_of_clusters - 1
            z[n] = -1  # ensures z[n] doesn't get counted as a cluster
            print 'z for now',z

            # unnormalized log probabilities for the clusters
            log_weights = np.nan * np.ones(number_of_clusters+1)
            
            # find the unnormalized log probabilities
            # for each existing cluster
            for cluster_id in range(number_of_clusters): 
                print 'cluster_id', cluster_id
                
                c_Precision = Prec0 + cluster_counts[cluster_id] * Prec
                c_Sig = np.linalg.inv(c_Precision)
                # find all of the points in this cluster
                loc_z = z==cluster_id
                # sum all the points in this cluster
                data_slice_c = data[loc_z]
                sum_data = data_slice_c.sum(axis=0)
                
                print 'sum_data', sum_data
                
                mat_prod1 = np.dot(Prec, sum_data)
                mat_prod2 = np.dot(Prec0, np.transpose(mu0))
                mat_sum = mat_prod1 + mat_prod2
                c_mean = np.dot(c_Sig, mat_sum)
                print 'c_mean',c_mean
                
                print 'cluster_counts[cluster_id]',cluster_counts[cluster_id]
                print 'np.log(cluster_counts[cluster_id])',np.log(cluster_counts[cluster_id])
                
                dmvn = np.log(dmvnorm(data[n,:],c_mean,c_Sig + Sig))
                print 'dmvn',dmvn
                log_weights[cluster_id] = np.log(cluster_counts[cluster_id]) + dmvn
                print 'log_weights updated',log_weights

            # find the unnormalized log probability
            # for the "new" cluster
            dmvn = np.log(dmvnorm(data[n,:],mu0,Sig0 + Sig))
            log_weights[number_of_clusters] = np.log(alpha) + dmvn
            print 'new log_weights',log_weights

            # transform unnormalized log probabilities
            # into probabilities
            max_weight = np.max(log_weights)
            log_weights = log_weights - max_weight
            loc_probs = np.exp(log_weights)
            loc_probs = loc_probs / np.sum(loc_probs)
            
            print 'loc_probs',loc_probs

            # sample which cluster this point should belong to
            newz = np.random.choice(range(number_of_clusters+1), replace=True, p=loc_probs)
            print 'newz',newz
            # if necessary, instantiate a new cluster
            if newz == number_of_clusters:
                cluster_counts.append(0)
                number_of_clusters = number_of_clusters + 1
            z[n] = newz
            # update the cluster counts
            cluster_counts[newz] = cluster_counts[newz] + 1
        if iteration >= minPauseIter:
            print z
    print 'final z',z
    return z
