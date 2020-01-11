import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp 


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

# Coordinates of the cross section
X1 = -0.1
Z1 = -0.1
X2 =  0.1
Z2 = -0.1
X3 =  0.1
Z3 =  0.1
X4 = -0.1
Z4 =  0.1



#General inputs required for the problem__________________________________________________
n_elem = int(input("Enter the number of elements: "))     # No of elements
per_elem = 2                                              # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                       # Total number of nodes 
Fixed_point = 0                                           # Coordinates of the beam
Free_point  = L
n_cross_elem = 2                                          # Number of L4 elements on each cross section
n_cross_nodes = 6                                         # Total number of nodes on each cross section
DOF = 3



# Mesh generation__________________________________________________________________________
# 1. Mesh generation along the beam axis(Y axis)___________________________________________
# coordinate = np.linspace(Fixed_point,Free_point,n_nodes)
# print(coordinate)
meshrefinementfactor = 5
q=meshrefinementfactor**(1/(n_elem-1))
l=(Fixed_point-Free_point)*(1-q)/(1-meshrefinementfactor*q)
rnode=Free_point
c=np.array([Free_point])
for i in range(n_elem):
    rnode=rnode+l
    c=np.append(c,rnode)
    l=l*q
coordinate = np.flip(c)

# 2. Mesh generation across the cross section (X,Z)______________________________________
a = np.linspace(X1,X4,2)
b = np.linspace(X2,X3,2)
c = np.linspace(Z1,Z4,3)
d = np.linspace(Z2,Z3,3)


#Along the beam axis(Y)__________________________________________________________________
xi =   0             #np.array([0.57735,-0.57735])                            # Gauss points
W_Length   =  2
Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])                 # Shape functions of a linear element
N_Der_xi   = np.array([-1/2,1/2])                              # Derivative of the shape function (N,xi)  



#Size of the global stiffness matrix and load vector computed using no of nodes and no of cross nodes on each node and DOF
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))
Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))    

for l in range(n_elem):
    
    J_Length = N_Der_xi@np.array([[coordinate[l]],            # Jacobian of each element along beam axis
                                 [coordinate[l+1]]])
    N_Der = np.array([-1/2*(1/J_Length),1/2*(1/J_Length)])    # Derivative of the shape functions with respect to physical coordinates (N,y)
    
    # Element stiffness matrix created using no of nodes per element and cross node and DOF  
    Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF)) 
    Elemental_Load_Vector = np.zeros((per_elem*n_cross_nodes*DOF,1))
    sep = int((per_elem*n_cross_nodes*DOF)/per_elem)          # Seperation point for stacking element stiffness matrix
    
    
    for i in range(len(Shape_func)):
        for j in range(len(Shape_func)):

            #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
            Global_Nodal_stiffness_matrix = np.zeros((n_cross_nodes*3,n_cross_nodes*3))
            Global_Nodal_Load_Vector =  np.zeros((n_cross_nodes*3,1))

            for m in range(n_cross_elem):                    # To loop over the cross sectional L4 elements (Over 2 elements for this case)
                X1 = a[0]
                X2 = b[0]
                X3 = b[1]
                X4 = a[1]
                Z1 = c[m]
                Z2 = d[m]
                Z3 = d[m+1]
                Z4 = c[m+1]
                # print(Z1,Z2,Z3,Z4)

                #Lagrange polynomials
                alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])           # Gauss points 
                beta  = np.array([0.57735,-0.57735,0.57735,-0.57735]) 
                W_Cs  = 1                                                       # weight for gauss quadrature in the cross section
                Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])


                #Lagrange Derivatives
                alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         # Derivatives of the lagrange polynomials
                beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     # with respect to alpha and beta
                X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
                X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
                Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
                Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4
                # print(X_alpha,X_beta,Z_alpha,Z_beta)


                J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)                      # Determinant of Jacobian matrix of the cross section
                # J_Cs = np.unique(J_Cs)
                # print(J_Cs)
                
                Nodal_stiffness_matrix = np.zeros((4*3,4*3))
                Nodal_Load_Vector = np.zeros((4*3,1))
                for tau_en,tau in enumerate(range(4)):
                    for s_en,s in enumerate(range(4)):
                    
                        # Fundamental nucleus of the stiffness matrix
                        # Derivative of F wrt to x and z for tau
                        F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
                        F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))
                        # Derivative of F wrt to x and z for s
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

                        
                        Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                        # if i == j == 1 and m==0:
                        #     Nodal_Load_Vector[2] = -6.25
                        #     Nodal_Load_Vector[5] = -6.25
                        #     Nodal_Load_Vector[8] = -18.75
                        #     Nodal_Load_Vector[11]= -18.75
                                                     
                        
                        # elif i == j == 1 and m==1:
                            
                        #     Nodal_Load_Vector[2] = -18.75
                        #     Nodal_Load_Vector[5] = -6.25
                        #     Nodal_Load_Vector[8] = -6.25
                        #     Nodal_Load_Vector[11]= -18.75
                        
                                          
                    
                A_c_e = np.zeros((12,18))             #Assignment matrix for assembling nodal stiffness matrix across the cross section
                np.fill_diagonal(A_c_e[0:9 , 6*m:9+6*m],1)
                np.fill_diagonal(A_c_e[9:12 , 15:18],1)
                A_c_e_T = np.transpose(A_c_e)

                K_Nodal = A_c_e_T@Nodal_stiffness_matrix@A_c_e
                Global_Nodal_stiffness_matrix = np.add(Global_Nodal_stiffness_matrix,K_Nodal)
                               
                Nodal_Load = A_c_e_T@Nodal_Load_Vector
                Global_Nodal_Load_Vector = np.add(Global_Nodal_Load_Vector,Nodal_Load)
            
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Global_Nodal_stiffness_matrix
            Elemental_Load_Vector[sep*i:sep*(i+1)]  = Global_Nodal_Load_Vector 
        # print(Global_Nodal_Load_Vector)    
    
    #Assignment matix for arranging global stiffness matrix
    A_fac = 18
    Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
    np.fill_diagonal( Ae[A_fac*0:A_fac*2 , A_fac*l:A_fac*(l+2)] , 1 )
    AeT = np.transpose(Ae)
    
    K = AeT@Elemental_stiffness_matrix@Ae
    Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

    K_Load = AeT@Elemental_Load_Vector
    Load_vector = np.add(Load_vector,K_Load)


# np.savetxt('Assignment_matrix.txt',np.ceil(A_c_e),delimiter=',')
# np.savetxt('2L4_Stiffness_matrix.txt',Global_stiffness_matrix,delimiter=',')
# print(Load_vector)

Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
Load_vector[n_nodes*n_cross_nodes*DOF-16]= 0
Load_vector[n_nodes*n_cross_nodes*DOF-13]= 0
Load_vector[n_nodes*n_cross_nodes*DOF-10]= -25
Load_vector[n_nodes*n_cross_nodes*DOF-7] = 0
Load_vector[n_nodes*n_cross_nodes*DOF-4] = 0
Load_vector[n_nodes*n_cross_nodes*DOF-1] = -25
# print(Load_vector)

Displacement = np.linalg.solve(Global_stiffness_matrix[18:,18:],Load_vector[18:])
print(Displacement)
print(Displacement.shape)
print(np.trace(Global_stiffness_matrix))


#For plotting z axis displacements
# Z_disp = np.array([])
# for k in range(n_nodes*n_cross_nodes-6):
#     Z_disp = np.append(Z_disp,Displacement[3*(k+1)-1])
# print(Z_disp)
# x_axis=np.arange(0,len(Z_disp),1)
# fig,ax = plt.subplots()
# ax.plot(x_axis,Z_disp)
# plt.show()