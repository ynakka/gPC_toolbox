#!/usr/bin/env python3
import numpy as np

def UnivariateGaussHermiteQuadrature(n):
    a = np.array([np.sqrt(i/2) for i in range(1,n)],dtype='float32')
    CM = np.diag(a,1) + np.diag(a,-1)

    (values,vectors) = np.linalg.eig(CM)
    indices = sorted(range(len(values)), key=lambda k: values[k])
    vectors = np.array([vectors[:,i] for i in indices])
    weight = np.power(vectors[:,0],2)
    node = np.array(sorted(values))*np.sqrt(2)
    return (weight, node)

if __name__ == "__main__":
    [weight, node] = UnivariateGaussHermiteQuadrature(20)
    print('WEIGHT')
    print(weight)
    print('NODE')
    print(node)