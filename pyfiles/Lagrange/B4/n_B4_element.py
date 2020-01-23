import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp 


#PARAMETERS
a = 0.2            #[m] Square cross section
L = 2              #[m] Length of the beam
E = 71.7e9           #[Pa] Young's Modulus
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
per_elem = 4                                              # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                       # Total number of nodes 
Fixed_point = 0                                           # Coordinates of the beam
Free_point  = L

#Mesh generation
coordinate = np.linspace(Fixed_point,Free_point,n_elem+1)
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
# print(coordinate)



#Along the beam axis(Y)
xi = np.array([0,0.5384693101056831,-0.5384693101056831,0.9061798459386640,-0.9061798459386640])                                                                     # Gauss points
W_Length   =   np.array([0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891])           
Shape_func = np.array([-9/16*(xi+1/3)*(xi-1/3)*(xi-1), 27/16*(xi+1)*(xi-1/3)*(xi-1),-27/16*(xi+1)*(xi+1/3)*(xi-1),9/16*(xi+1/3)*(xi-1/3)*(xi+1)])                       # Shape functions
N_Der_xi = N_Der_xi = np.array([-1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') + 0.0625,5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') - 1.6875,-5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') + 1.6875,1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') - 0.0625])    # Derivative of the shape function (N,xi)
N_Der_xi_m = np.array([0.0625,- 1.6875,1.6875,-0.0625])   



#Along the Beam cross section (X,Z)
#Lagrange polynomials
alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])                      # Gauss points 
beta  = np.array([0.57735,-0.57735,0.57735,-0.57735]) 
W_Cs  = 1                                                                  # weight for gauss quadrature in the cross section
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])
n_cross_nodes = len(Lag_poly)                                              # No of lagrange nodes per node
DOF = 3                                                                    # Degree of freedom of each lagrange node

#Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         # Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     # with respect to alpha and beta
X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4


J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)                                              # Determinant of Jacobian matrix of the cross section
J_Cs = np.unique(J_Cs)


#Size of the global stiffness matrix computed using no of nodes and no of cross nodes on each node and DOF
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))    
for l in range(n_elem):
    # print(l)
    mid = (coordinate[l+1]+coordinate[l])/2
    mid_length = (coordinate[l+1]-coordinate[l])/2 
    X_coor = np.array([[coordinate[l]],
                      [mid-(mid_length/3)], 
                      [mid+(mid_length/3)],
                      [coordinate[l+1]]])                                                      

    # print(X_coor)               
    J_Length   = N_Der_xi_m@X_coor                                            # Jacobian for the length of the beam
    N_Der      = np.array([(-1.6875*xi**2 + 1.125*xi + 0.0625)*(1/J_Length),(5.0625*xi**2 - 1.125*xi - 1.6875)*(1/J_Length),(-5.0625*xi**2 - 1.125*xi + 1.6875)*(1/J_Length),(1.6875*xi**2 + 1.125*xi - 0.0625)*(1/J_Length)])        # Derivative of the shape function wrt to physical coordinates(N,y)

    
    
    # Element stiffness matrix created using no of nodes per element and cross node and DOF  
    Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF)) 
    sep = int((per_elem*n_cross_nodes*DOF)/per_elem)                             # Seperation point for stacking element stiffness matrix                  

    for i in range(len(Shape_func)):
        
        for j in range(len(Shape_func)):
            #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
            Nodal_stiffness_matrix = np.zeros((n_cross_nodes*DOF,n_cross_nodes*DOF))
            
            for tau_en,tau in enumerate(range(n_cross_nodes)):
                for s_en,s in enumerate(range(n_cross_nodes)):
                    
                    #Fundamental nucleus of the stiffness matrix
                    #Derivative of F wrt to x and z for tau
                    F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
                    F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))

                    #Derivative of F wrt to x and z for s
                    F_s_x = 1/J_Cs*((Z_beta*alpha_der[s])-(Z_alpha*beta_der[s]))
                    F_s_z = 1/J_Cs*((-X_alpha*alpha_der[s])+(X_beta*beta_der[s]))
                    # print(Shape_func[i])
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
                    # print(F_Nu)
                    Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                    
            
                    
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
        
    #Assignment matix for arranging global stiffness matrix
    A_fac = 12
    Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))
    np.fill_diagonal( Ae[A_fac*0:A_fac*4 , A_fac*3*l:A_fac*(4+(3*l))] , 1 )
    AeT = np.transpose(Ae)
    
    
    K = AeT@Elemental_stiffness_matrix@Ae
    Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

print(Global_stiffness_matrix.shape)               
np.savetxt('B4_Stiffness_matrix.txt',Global_stiffness_matrix,delimiter=',')
# np.savetxt('B3_Stiffness_matrix_size.txt',Global_stiffness_matrix.shape,delimiter=',')



Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
Load_vector[n_nodes*n_cross_nodes*DOF-10]= -12.5
Load_vector[n_nodes*n_cross_nodes*DOF-7] = -12.5
Load_vector[n_nodes*n_cross_nodes*DOF-4] = -12.5
Load_vector[n_nodes*n_cross_nodes*DOF-1] = -12.5
print("Load vector ----------------------------------------------")
print(Load_vector)




Displacement = np.linalg.solve(Global_stiffness_matrix[12:,12:],Load_vector[12:])
print('Displacement-----------------------------------------------------')
print(Displacement)
print(Displacement.shape)
# np.savetxt('n_B4_Displacement.txt',Displacement)
print(np.linalg.norm(Global_stiffness_matrix))


# Z_disp = np.array([])
# for k in range(n_nodes*n_cross_nodes-4):
#     Z_disp = np.append(Z_disp,Displacement[3*(k+1)-1])
# print(Z_disp.shape)
# x_axis=np.arange(0,len(Z_disp),1)
# fig,ax = plt.subplots()
# ax.plot(x_axis,Z_disp)
# plt.show()
Z_disp = np.array([])

for k in range(n_nodes-1):
    Z_disp = np.append(Z_disp,np.unique(Displacement[12*(k+1)-1]))
# print(np.unique(Z_disp))
x_axis=np.arange(0,len(Z_disp),1)
fig,ax = plt.subplots()
ax.plot(x_axis,Z_disp)
plt.show()