import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp 


#PARAMETERS
h = 1              #Height of the component
L = 3              #[m] Length of the beam
E = 75e9           #[Pa] Young's Modulus
v = 0.33           #Poissons Ratio
F_Z = -10**4       #[N] Load applied


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
#Lower Flange coordinates
X1 = -0.015
Z1 = -0.5
X2 =  0.015
Z2 = -0.5
X3 =  0.015
Z3 = -0.47
X4 = -0.015
Z4 = -0.47


X_coor_fl1 = np.array([X1,X2,X3,X4])
Z_coor_fl1 = np.array([Z1,Z2,Z3,Z4])



#Web coordinates
X5 = -0.0005
Z5 = -0.47
X6 =  0.0005
Z6 = -0.47
X7 =  0.0005
Z7 =  0.47
X8 = -0.0005
Z8 =  0.47



X_coor_w = np.array([X5,X6,X7,X8])
Z_coor_w = np.array([Z5,Z6,Z7,Z8])

#Upper flange coordinates
X9  = -0.015
Z9  =  0.5
X10 =  0.015
Z10 =  0.5
X11 =  0.015
Z11 =  0.47
X12 = -0.015
Z12 =  0.47
X_coor_fl2 =  np.array([X9,X10,X11,X12])
Z_coor_fl2 =  np.array([Z9,Z10,Z11,Z12])


#Coordinate array----Cross sectional coordinates
X_c_coor = np.array([X_coor_fl1,X_coor_w,X_coor_fl2])
Z_c_coor = np.array([Z_coor_fl1,Z_coor_w,Z_coor_fl2])


#General inputs required for the problem__________________________________________________
n_elem = int(input("Enter the number of elements: "))     # No of elements
per_elem = 3                                              # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                       # Total number of nodes 
Fixed_point = 0                                           # Coordinates of the beam
Free_point  = L
n_cross_elem = 3                                          # Number of L4 elements on each cross section
n_cross_nodes = 12                                         # Total number of nodes on each cross section
DOF = 3

# Mesh generation__________________________________________________________________________
# 1. Mesh generation along the beam axis(Y axis)___________________________________________
coordinate = np.linspace(Fixed_point,Free_point,n_elem+1)
# print(coordinate)
# meshrefinementfactor = 5
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
xi = np.array([0.339981,-0.339981,0.861136,0.861136])                                                   # Gauss points
W_Length   = np.array([0.652145,0.652145,0.347855,0.347855])  
Shape_func = np.array([1/2*(xi**2-xi),1-xi**2,1/2*(xi**2+xi)])                       # Shape functions
N_Der_xi = np.array([sp.Symbol('xi')-1/2,-2*sp.Symbol('xi'),sp.Symbol('xi')+1/2])    # Derivative of the shape function 
N_Der_xi_m = np.array([-1/2,1/2])             # Taking just numerical values from shape function for easily computing jacobian

#Size of the global stiffness matrix and load vector computed using no of nodes and no of cross nodes on each node and DOF
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))
Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1)) 

for l in range(n_elem):

    X_coor = np.array([[coordinate[l]],
                       [coordinate[l+1]]])                          #[(coordinate[l]+coordinate[l+1])/2],                              
    J_Length   = round(np.asscalar(N_Der_xi_m@X_coor),4)                                             # Jacobian for the length of the beam
    N_Der      = np.array([(xi-1/2)*(1/J_Length),-2*xi*(1/J_Length),(xi+1/2)*(1/J_Length)])                         # Derivative of the shape function wrt to physical coordinates(N,y)
    
    # Element stiffness matrix created using no of nodes per element and cross node and DOF  
    Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF)) 
    sep = int((per_elem*n_cross_nodes*DOF)/per_elem)                             # Seperation point for stacking element stiffness matrix                  


    for i in range(len(Shape_func)):
        for j in range(len(Shape_func)):

            #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
            Global_Nodal_stiffness_matrix = np.zeros((n_cross_nodes*3,n_cross_nodes*3))
            Global_Nodal_Load_Vector =  np.zeros((n_cross_nodes*3,1))

            for m in range(n_cross_elem):                    # To loop over the cross sectional L4 elements (Over 3 elements for this case)
                X1,X2,X3,X4 =  X_c_coor[m]
                Z1,Z2,Z3,Z4 =  Z_c_coor[m]
                
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
                J_Cs = J_Cs[0]
                # print(J_Cs)
                
                
                Nodal_stiffness_matrix = np.zeros((4*3,4*3))
                Nodal_Load_Vector = np.zeros((4*3,1))
                for tau_en,tau in enumerate(range(4)):
                    for s_en,s in enumerate(range(4)):
                    
                        # Fundamental nucleus of the stiffness matrix
                        # Derivative of F wrt to x and z for tau
                        # print((Z_alpha*beta_der[tau]))
                        
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


                        # if (i==j==0) and (tau == s) and (l==0):
                    
                        #     np.fill_diagonal(F_Nu,30e12)
                        # if (i==j==1) and (tau==s):
                        #     F_Nu[0,0] = 30e12
                        # if (i==j==2) and (tau==s):
                        #     F_Nu[0,0] = 30e12
                        Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu

                A_c_fac = 12
                A_c_e = np.zeros((12,36))
                np.fill_diagonal(A_c_e[0:A_c_fac , A_c_fac*m:A_c_fac*(m+1)],1)
                A_c_e_T = np.transpose(A_c_e)

                K_Nodal = A_c_e_T@Nodal_stiffness_matrix@A_c_e
                Global_Nodal_stiffness_matrix = np.add(Global_Nodal_stiffness_matrix,K_Nodal)

            
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Global_Nodal_stiffness_matrix
            # Elemental_Load_Vector[sep*i:sep*(i+1)]  = Global_Nodal_Load_Vector 


    #Assignment matix for arranging global stiffness matrix
    A_fac = 36
    Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
    np.fill_diagonal( Ae[A_fac*0:A_fac*3 , A_fac*2*l:A_fac*(3+(2*l))] , 1 )
    AeT = np.transpose(Ae)
    
    K = AeT@Elemental_stiffness_matrix@Ae
    Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)



print(Global_stiffness_matrix)
print(Global_stiffness_matrix.shape)


Load_vector[n_nodes*n_cross_nodes*DOF-22] = -2500
Load_vector[n_nodes*n_cross_nodes*DOF-19] = -2500
Load_vector[n_nodes*n_cross_nodes*DOF-16] = -2500
Load_vector[n_nodes*n_cross_nodes*DOF-13] = -2500


Displacement = np.linalg.solve(Global_stiffness_matrix[36:,36:],Load_vector[36:])
print(Displacement)
print(Displacement.shape)