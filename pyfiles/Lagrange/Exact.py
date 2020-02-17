
import numpy as np 

L = 2
E = 75e9
a = 0.2
I = a**4/12
b = E*I/(L**3)
K = b*np.array([[12,6*L,-12,6*L],
              [6*L,4*L*L,-6*L,2*L*L],
              [-12,-6*L,12,-6*L],
              [6*L,2*L*L,-6*L,4*L*L]])
    

d = np.zeros((4,1))
d[2] = -50
print(d)
ans = np.linalg.solve(K[2:,2:],d[2:])
print(ans)
print(np.linalg.norm(K))
