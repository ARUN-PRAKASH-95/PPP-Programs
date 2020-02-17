import numpy as np
import matplotlib.pyplot as pyplot
from sympy import *
x = symbols('x')
z = symbols('z')


#PARAMETERS
low = 0
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
# print(C_22,C_66,C_44)

S_point = 18


#Along the beam axis(Y)
n_elem = 1                                               # No of elements
per_elem = 3                                             # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                      # Total no of nodes 
xi = np.array([0,0.774597,-0.774597])             #np.array([0.57735,-0.57735])                        # Gauss points
W_Length   = np.array([0.888889,0.555556,0.555556])                                        # Weight for the gauss quadrature
Shape_func = np.array([1/2*(xi**2-xi),1-xi**2,1/2*(xi**2+xi)])                #np.array([1/2*(1-xi),1/2*(1+xi)])
N_Der_xi   = np.array([symbols('xi')-1/2,-2*symbols('xi'),symbols('xi')+1/2])                #np.array([-1/2,1/2])                        # Derivative of the shape function (N,xi)

#Things that change by changing number of elements
X_coor     = np.array([[0], 
                      [L/2],
                      [L]])
J_Length   = N_Der_xi@X_coor                                                  # Jacobian for the length of the beam
N_Der      = np.array([(xi-1/2),-2*xi,(xi+1/2)]) #np.array([-1/2*(1/J_Length),1/2*(1/J_Length)])                   # Derivative of the shape function (N,y)
print(J_Length)


#Along the Beam cross section (X,Z)
#Taylor polynomials N=1
Taylor_poly = np.array([1,x,z,x**2,x*z,z**2])
X_der = np.array([0,1,0,2*x,z,0])
Z_der = np.array([0,0,1,0,x,2*z])
n_cross_nodes = len(Taylor_poly)
DOF=3


Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF))
sep = int((per_elem*n_cross_nodes*DOF)/per_elem)                             # Seperation point for stacking element stiffness matrix 


for i in range(per_elem):
    
    for j in range(per_elem):
            #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
        Nodal_stiffness_matrix = np.zeros((n_cross_nodes*DOF,n_cross_nodes*DOF))
        for tau_en,tau in enumerate(range(n_cross_nodes)):
            for s_en,s in enumerate(range(n_cross_nodes)):
                
                K_xx =  C_22*integrate(X_der[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(Z_der[tau]*Z_der[s],(x,low,a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,low,a),(z,low,a))*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                K_xy =  C_23*integrate(Taylor_poly[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_44*integrate(X_der[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)  
                K_xz =  C_12*integrate(Z_der[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(X_der[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length)     
                K_yx =  C_44*integrate(Taylor_poly[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_23*integrate(X_der[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                K_yy =  C_55*integrate(Z_der[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*integrate(X_der[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_33*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length) 
                K_yz =  C_55*integrate(Taylor_poly[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_13*integrate(Z_der[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                K_zx =  C_12*integrate(X_der[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(Z_der[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length)
                K_zy =  C_13*integrate(Taylor_poly[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_55*integrate(Z_der[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)  
                K_zz =  C_11*integrate(Z_der[tau]*Z_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*integrate(X_der[tau]*X_der[s],(x,low, a),(z,low,a))*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_55*integrate(Taylor_poly[tau]*Taylor_poly[s],(x,low, a),(z,low,a))*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                F_Nu = np.array([[K_xx,K_xy,K_xz],[K_yx,K_yy,K_yz],[K_zx,K_zy,K_zz]])
                # print(tau,F_Nu)
                
                if (i==j==0) and (tau == s):
                    np.fill_diagonal(F_Nu,30e12)
                Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                
                
              
                
        
                
        Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix

#Stiffness matrix checkers
# print("Transpose",np.allclose(Elemental_stiffness_matrix,Elemental_stiffness_matrix.T))
# print("Inverse",np.linalg.inv(Elemental_stiffness_matrix))
# print("Determinant",np.linalg.det(Elemental_stiffness_matrix))
# EV,EVector = np.linalg.eig(Elemental_stiffness_matrix)
# print("Eigen_value",EV)




# print(Elemental_stiffness_matrix.shape)
# print("Stiffness matrix ----------------------------------------")
# print(Elemental_stiffness_matrix) 
# # print(Elemental_stiffness_matrix[12,3])               



Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
Load_vector[n_nodes*n_cross_nodes*DOF-16] = -50
Load_vector[n_nodes*n_cross_nodes*DOF-13] = -5
Load_vector[n_nodes*n_cross_nodes*DOF-10] = -5
Load_vector[n_nodes*n_cross_nodes*DOF-7] = -0.5
Load_vector[n_nodes*n_cross_nodes*DOF-4] = -0.5
Load_vector[n_nodes*n_cross_nodes*DOF-1] = -0.5
# print("Load vector ----------------------------------------------")
# print(Load_vector)

C = np.linalg.solve(Elemental_stiffness_matrix[S_point:,S_point:],Load_vector[S_point:])
print("Displacement----------------------------------------------")
print(C)
# # np.savetxt('Taylor_stiffness.txt',Elemental_stiffness_matrix,delimiter=',')
# print(C.shape)