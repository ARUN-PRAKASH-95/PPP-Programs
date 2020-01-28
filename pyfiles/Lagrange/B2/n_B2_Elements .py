import numpy as np
import matplotlib.pyplot as plt  
from sympy import *
from sympy.solvers.solveset import linsolve



#PARAMETERS
a = 0.2            #[m] Square cross section
L = 2              #[m] Length of the beam
E = 75e9           #[Pa] Young's Modulus
v = 0.33           #Poissons Ratio
G = E/(2*(1+v))
First  = (E*(1-v))/((1+v)*(1-2*v))
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


n_elem = int(input("Enter the number of elements: "))     # No of elements
per_elem = 2                                              # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                       # Total number of nodes 
Fixed_point = 0                                           # Coordinates of the beam
Free_point  = 2


# Mesh generation_________________________________________
# 1. Mesh generation along the beam axis(Y axis) 
coordinate = np.linspace(Fixed_point,Free_point,n_elem+1)
# print(coordinate)
# meshrefinementfactor = 2
# q=meshrefinementfactor**(1/(n_elem-1))
# l=(Fixed_point-Free_point)*(1-q)/(1-meshrefinementfactor*q)
# rnode=Free_point
# c=np.array([Free_point])
# for i in range(n_elem):
#     rnode=rnode+l
#     c=np.append(c,rnode)
#     l=l*q
# coordinate = np.flip(c)


#Along the beam axis(Y)
xi = 0#np.array([0.57735,-0.57735])                              # Gauss points
W_Length   = 2
Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])                 # Shape functions of a linear element
N_Der_xi   = np.array([-1/2,1/2])                              # Derivative of the shape function (N,xi)  


#Along the Beam cross section (X,Z)
#Lagrange polynomials
alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])           # Gauss points 
beta  = np.array([0.57735,-0.57735,0.57735,-0.57735]) 
W_Cs  = 1                                                       # weight for gauss quadrature in the cross section
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])
n_cross_nodes = len(Lag_poly)                                             # No of lagrange nodes per node
DOF = 3                                                            # Degree of freedom of each lagrange node

#Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         # Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     # with respect to alpha and beta

X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4


J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)                      # Determinant of Jacobian matrix of the cross section
J_Cs = np.unique(J_Cs)


#Size of the global stiffness matrix computed using no of nodes and no of cross nodes on each node and DOF
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))    
for l in range(n_elem):
    
    J_Length = N_Der_xi@np.array([[coordinate[l]],            # Jacobian of each element along beam axis
                                [coordinate[l+1]]])
    # print(J_Length)
    # Derivative of the shape functions with respect to physical coordinates (N,y)
    N_Der = np.array([-1/2*(1/J_Length),1/2*(1/J_Length)])  
    
    
    # Element stiffness matrix created using no of nodes per element and cross node and DOF  
    Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF)) 
    sep = int((per_elem*n_cross_nodes*DOF)/per_elem)                             # Seperation point for stacking element stiffness matrix                  

    for i in range(len(Shape_func)):
        for j in range(len(Shape_func)):
            #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
            Nodal_stiffness_matrix = np.zeros((n_cross_nodes*3,n_cross_nodes*3))
            for tau_en,tau in enumerate(range(n_cross_nodes)):
                for s_en,s in enumerate(range(n_cross_nodes)):
                    
                    #Fundamental nucleus of the stiffness matrix
                    #Derivative of F wrt to x and z for tau
                    F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
                    F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))

                    #Derivative of F wrt to x and z for s
                    F_s_x = 1/J_Cs*((Z_beta*alpha_der[s])-(Z_alpha*beta_der[s]))
                    F_s_z = 1/J_Cs*((-X_alpha*alpha_der[s])+(X_beta*beta_der[s]))
                    
                    
                    K_xx =  C_22*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                    K_xy =  C_23*np.sum(W_Cs*Lag_poly[tau]*F_s_x*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*F_tau_x*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_xz =  C_12*np.sum(W_Cs*F_tau_z*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_x*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length)     
                    K_yx =  C_44*np.sum(W_Cs*Lag_poly[tau]*F_s_x*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_23*np.sum(W_Cs*F_tau_x*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_yy =  C_55*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_33*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length) 
                    K_yz =  C_55*np.sum(W_Cs*Lag_poly[tau]*F_s_z*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_13*np.sum(W_Cs*F_tau_z*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_zx =  C_12*np.sum(W_Cs*F_tau_x*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_z*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) 
                    K_zy =  C_13*np.sum(W_Cs*Lag_poly[tau]*F_s_z*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_55*np.sum(W_Cs*F_tau_z*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)  
                    K_zz =  C_11*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_55*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                    F_Nu = np.array([[K_xx,K_xy,K_xz],[K_yx,K_yy,K_yz],[K_zx,K_zy,K_zz]])
                    
                    
                    if (i==j==0) and (tau == s) and (l==0):
                    # print(F_Nu)
                        np.fill_diagonal(F_Nu,30e12)
                    if (i==j==1) and (tau==s):
                        F_Nu[0,0] = 30e12
                        # F_Nu[2,2] = 30e12
                    Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                    
            
                    
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
        
    #Assignment matix for arranging global stiffness matrix
    A_fac = 12
    Ae = np.zeros((per_elem*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
    np.fill_diagonal( Ae[0:A_fac*2 , A_fac*l:A_fac*(l+2)] , 1 )
    # np.fill_diagonal( Ae[A_fac*1:A_fac*2 , A_fac*(l+1):A_fac*(l+2)] , 1 )
    AeT = np.transpose(Ae)
    # print(Elemental_stiffness_matrix)
    K = AeT@Elemental_stiffness_matrix@Ae
    Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)
# print(Global_stiffness_matrix)    
               

Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
# Load_vector[n_nodes*n_cross_nodes*DOF-10] = -12.5
# Load_vector[n_nodes*n_cross_nodes*DOF-7]  = -12.5
# Load_vector[n_nodes*n_cross_nodes*DOF-4]  = -12.5
# Load_vector[n_nodes*n_cross_nodes*DOF-1]  = -12.5




Load_vector[n_nodes*n_cross_nodes*DOF-11] = 12.5
Load_vector[n_nodes*n_cross_nodes*DOF-8]  = 12.5
Load_vector[n_nodes*n_cross_nodes*DOF-5]  = 12.5
Load_vector[n_nodes*n_cross_nodes*DOF-2]  = 12.5
print("Load vector ----------------------------------------------")
# print(Load_vector)

Displacement = np.linalg.solve(Global_stiffness_matrix,Load_vector)
print(Displacement)


Z_disp = np.array([])

# for k in range(n_nodes-1):
#     Z_disp = np.append(Z_disp,np.unique(Displacement[12*(k+1)-1]))
# # print(np.unique(Z_disp))
# x_axis=np.arange(0,len(Z_disp),1)
# fig,ax = plt.subplots()
# ax.plot(x_axis,Z_disp)
# plt.show()


#To extract the displacement of our interest 

#X displacements of all the lagrange nodes
X_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    X_disp = np.append(X_disp,Displacement[3*(k+1)-3])
Req_X_disp = X_disp[-4::]                       #Displacement of the lagrange nodes at end cross section


#Y displacements of all the lagrange nodes
Y_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    Y_disp = np.append(Y_disp,Displacement[3*(k+1)-2])
print(Y_disp)
x_axis=np.arange(0,len(Y_disp),1)
Req_Y_disp = Y_disp[-4::]


#Z displacements of all the lagrange nodes
Z_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    Z_disp = np.append(Z_disp,Displacement[3*(k+1)-1])
print(Z_disp)
Req_Z_disp = Z_disp[-4::]






#Post processing
alpha,beta = symbols('alpha,beta')
F1 = 1/4*(1-alpha)*(1-beta)
F2 = 1/4*(1+alpha)*(1-beta)
F3 = 1/4*(1+alpha)*(1+beta)
F4 = 1/4*(1-alpha)*(1+beta)

X1 = -0.1
Z1 = -0.1
X2 =  0.1
Z2 = -0.1
X3 =  0.1
Z3 =  0.1
X4 = -0.1
Z4 =  0.1

X = np.array([-0.1])
Z = np.array([-0.1])

#Loop for finding the natural coordinates of the physical domain
for i in range(len(X)):
    eq1 =  F1*X1 + F2 * X2 + F3 * X3 + F4 * X4 - X[i]
    eq2 =  F1*Z1 + F2 * Z2 + F3 * Z3 + F4 * Z4 - Z[i]
    a = solve([eq1, eq2], (alpha,beta))
    print(a)


#Natural coordinates of the points in the physical domain
X_nat = a[alpha]
Y_nat = a[beta]
Lag_poly = np.array([1/4*(1-X_nat)*(1-Y_nat),1/4*(1+X_nat)*(1-Y_nat),1/4*(1+X_nat)*(1+Y_nat),1/4*(1-X_nat)*(1+Y_nat)])



#Axial strain
Epsilon_yy =  Lag_poly[0]*1/2*(1/J_Length)*Req_Y_disp[0] + Lag_poly[1]*1/2*(1/J_Length)*Req_Y_disp[1] + Lag_poly[2]*1/2*(1/J_Length)*Req_Y_disp[2] + Lag_poly[3]*1/2*(1/J_Length)*Req_Y_disp[3] 
print(Epsilon_yy)


fig,ax = plt.subplots()
ax.plot(x_axis,Y_disp)
plt.show()