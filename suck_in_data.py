#SUCK PYTHON DATA IN AND CREATE FEATURE VECTORS

import json
import numpy
from pprint import pprint

with open('python_data/compDeriv/solutions.json') as solutions_file:    
    solutions = json.load(solutions_file)

with open('python_data/compDeriv/variables.json') as variables_file:    
    variables = json.load(variables_file)

#pprint(solutions)
#pprint(variables)
#pprint(solutions[100])
#pprint(solutions[1000])
#pprint(solutions[2000])
#pprint(solutions[3000])
num_of_solutions = len(solutions)
num_of_abstract_variables = len(variables)

print 'num_of_solutions, num_of_abstract_variables',num_of_solutions, num_of_abstract_variables

#iterate through solutions
feature_vectors = numpy.zeros((num_of_solutions,num_of_abstract_variables))
for i, sol in enumerate(solutions):
    #produce a feature vector of length = number of abstract variables
    #sol.append(numpy.zeros(num_of_abstract_variables))
    print i, sol
    feature_vector_for_sol_i = feature_vectors[i]
    for abstract_var_id in sol['variableIDs']:
        feature_vector_for_sol_i[abstract_var_id-1] = 1 #var_id is one-indexed; feature vector isn't
    print feature_vector_for_sol_i
    break

