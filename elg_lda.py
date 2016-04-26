# Elena's implementation of LDA

#based loosely on https://gist.github.com/mblondel/542786

import numpy as np
import scipy as sp
from scipy.special import gammaln

#Just some initialization; these can be changed, of course
n_topics = 10
alpha = 0.1
beta = 0.1
maxiteration = 10

#a matrix of #docs vs. #vocab
doc_matrix = np.matrix('1 2; 3 4') #TODO: REPLACE WITH FAUX DATA
n_docs, n_words = doc_matrix.shape


##Initialization
n_dz = np.zeros((n_docs,n_topics))
n_zw = np.zeros((n_topics,n_words))

n_d = np.zeros(n_docs)
n_z = np.zeros(n_topics)

topics = {}

for d in xrange(n_docs):
    for w in [word_index in doc_matrix[d,:] if word_index !== -1]:
        for w in n_vocab:
            z = np.random.randint(n_topics)
            n_dz[d,z] += 1
            n_d[d] += 1
            n_zw[z,w] += 1
            n_z += 1
            topics[(d,w)] = z

def conditional_dist(d,w,n_zw,n_z,n_dz,n_d):
    global alpha
    global beta

    n_words = n_zw.shape[1]

    #todo: understand this formula better
    numerator1 = n_zw[:,w] + beta #
    denominator1 = n_z + beta*n_words

    numerator2 = n_dz[d,:] + alpha
    denominator2 = n_d + alpha*n_words

    p_z = numerator1/denominator1 * numerator2/denominator2
    p_z /= np.sum(p_z)

    return p_z

##Run Gibbs
for iteration in xrange(maxiteration):
    for d in xrange(n_docs):
        for w in [word_index in doc_matrix[d,:] if word_index !== -1]:
            z = topics[(d,w)]

            #remove assignment
            n_dz[d,z] -= 1
            n_d[d] -= 1
            n_zw[z,w] -= 1
            n_z -= 1

            #compute probability over topics of held-out assignment
            p_z = conditional_dist(d,w,n_zw,n_z,n_dz,n_d)

            #get new assignment by sampling from p_z
            z = np.random.multinomial(1,p_z).argmax()

            #add new assignment in the place of the old assignment
            n_dz[d,z] += 1
            n_d[d] += 1
            n_zw[z,w] += 1
            n_z += 1


#grabbed straight from https://gist.github.com/mblondel/542786 for testing:

N_TOPICS = 10
DOCUMENT_LENGTH = 100


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def gen_word_distribution(n_topics, document_length):
    """
    Generate a word distribution for each of the n_topics.
    """
    width = n_topics / 2
    vocab_size = width ** 2
    m = np.zeros((n_topics, vocab_size))

    for k in range(width):
        m[k,:] = vertical_topic(width, k, document_length)

    for k in range(width):
        m[k+width,:] = horizontal_topic(width, k, document_length)

    m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

    return m

def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH, alpha=0.1):
    """
    Generate a document:
        1) Sample topic proportions from the Dirichlet distribution.
        2) Sample a topic index from the Multinomial with the topic
           proportions from 1).
        3) Sample a word from the Multinomial corresponding to the topic
           index from 2).
        4) Go to 2) if need another word.
    """
    theta = np.random.mtrand.dirichlet([alpha] * n_topics)
    v = np.zeros(vocab_size)
    for n in range(length):
        z = sample_index(theta)
        w = sample_index(word_dist[z,:])
        v[w] += 1
    return v

def gen_documents(word_dist, n_topics, vocab_size, n=500):
    """
    Generate a document-term matrix.
    """
    m = np.zeros((n, vocab_size))
    for i in xrange(n):
        m[i, :] = gen_document(word_dist, n_topics, vocab_size)
    return m

width = N_TOPICS / 2
vocab_size = width ** 2
word_dist = gen_word_distribution(N_TOPICS, DOCUMENT_LENGTH)
matrix = gen_documents(word_dist, N_TOPICS, vocab_size)