import numpy as np
import matplotlib.pyplot as pyplot
from sympy import *
x = symbols('x')
z = symbols('z')


#PARAMETERS
a = 0.2            #[m] Square cross section
L = 2              #[m] Length of the beam
E = 75e9           #[Pa] Young's Modulus
v = 0.33           #Poissons Ratio
G = E/(2*(1+v))
First  = E*(1-v)/((1+v)*(1-2*v))
Second = v*E/((1+v)*(1-2*v))
C_11 = First
C_22 = First
C_33 = First
C_12 = Second
C_13 = Second
C_23 = Second
C_44 = G
C_55 = G
C_66 = G
#print(C_22,C_66,C_44)
#Coordinates of the cross section
X1 = -0.1
Z1 = -0.1
X2 =  0.1
Z2 = -0.1
X3 =  0.1
Z3 =  0.1
X4 = -0.1
Z4 =  0.1

#Along the beam axis(Y)
n_elem = 1                                             # No of elements
n  = 2                                                 # No of nodes 
xi = np.array([0.57735,-0.57735])                      # Gauss points
Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])
N_Der_xi      = np.array([-1/2,1/2])                   # Derivative of the shape function (N,xi)
W_Length   =  1                                        # Weight for the gauss quadrature

#Things that change by changing number of elements
X_coor     = np.array([0,L])
J_Length   = N_Der_xi@X_coor                                                     #Jacobian for the length of the beam
N_Der      = np.array([-1/2*(1/J_Length),1/2*(1/J_Length)])                   #Derivative of the shape function (N,xi)



#Along the Beam cross section (X,Z)
#Taylor polynomials N=1
T_poly = 3
Taylor_poly = np.array([1,x,z])
X_der = np.array([0,1,0])
Z_der = np.array([0,0,1])

Elemental_stiffness_matrix = np.zeros((n*T_poly*3,n*T_poly*3))
sep = int((n*T_poly*3)/n)        #Seperation point for stacking element stiffness matrix 

for i in range(n):
    for j in range(n):
        #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
        Nodal_stiffness_matrix = np.zeros((T_poly*3,T_poly*3))
        for tau_en,tau in enumerate(range(T_poly)):
            for s_en,s in enumerate(range(T_poly)):
                
                K_xx =  C_22*integrate(X_der[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(Z_der[tau]*Z_der[s] ,(x,-0.1,0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_44*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,-0.1,0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length)
                K_xy =  C_23*integrate(Taylor_poly[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length) + C_44*integrate(X_der[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length)  
                K_xz =  C_12*integrate(Z_der[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(X_der[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)     
                K_yx =  C_44*integrate(Taylor_poly[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length) + C_23*integrate(X_der[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length)
                K_yy =  C_55*integrate(Z_der[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_44*integrate(X_der[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_33*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length) 
                K_yz =  C_55*integrate(Taylor_poly[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length) + C_13*integrate(Z_der[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length)
                K_zx =  C_12*integrate(X_der[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(Z_der[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)
                K_zy =  C_13*integrate(Taylor_poly[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length) + C_55*integrate(Z_der[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length)  
                K_zz =  C_11*integrate(Z_der[tau]*Z_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(X_der[tau]*X_der[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length) + C_55*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,-0.1, 0.1),(z,-0.1,0.1))*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length)
                F_Nu = np.array([[K_xx,K_xy,K_xz],[K_yx,K_yy,K_yz],[K_zx,K_zy,K_zz]])
                # if (i==j==0) and (tau == s):
                #     np.fill_diagonal(F_Nu,30e12)
                Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                
                
              
        #print(Nodal_stiffness_matrix)         
                
        
                
        Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
print(Elemental_stiffness_matrix.shape)
print("Stiffness matrix ----------------------------------------")
print(Elemental_stiffness_matrix[3,14])                
Load_vector = np.zeros((n*T_poly*3,1))
Load_vector[11] = -50
print("Load vector ----------------------------------------------")
print(Load_vector)

C = np.linalg.solve(Elemental_stiffness_matrix[9:,9:],Load_vector[9:])
print("Displacement----------------------------------------------")
print(C)
