import numpy as np 

L = 2 
b = 1249687.5

K = b*np.array([[12,6*L,-12,6*L],
              [6*L,4*L*L,-6*L,2*L*L],
              [-12,-6*L,12,-6*L],
              [6*L,2*L*L,-6*L,4*L*L]])
    
print(K)
d = np.zeros((4,1))
d[2] = -50
print(d)
ans = np.linalg.solve(K[2:,2:],d[2:])
print(ans)