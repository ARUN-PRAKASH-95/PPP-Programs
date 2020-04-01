                                           
                                                ####### TEST CASE 2 #######

# OBJECTIVE:
# To Compare one of the components(15*3) of the Element stiffness matrix with the analytical solution given in the literature #
 

import numpy as np


############################ MATERIAL AND GEOMETRY PARAMETERS  #############################

a = 0.2            #[m] Square cross section
b = a
L = 2              #[m] Length of the beam
E = 75e9           #[Pa] Young's Modulus
v = 0.33           #Poissons Ratio
G = E/(2*(1+v))
First  = (E*(1-v))/((1+v)*(1-2*v))
Second = v*E/((1+v)*(1-2*v))

#____Elastic constants____#
C_11 = First
C_22 = First
C_33 = First
C_12 = Second
C_13 = Second
C_23 = Second
C_44 = G
C_55 = G
C_66 = G



#____________Coordinates of the cross section____________#
X1 = -0.1
Z1 = -0.1
X2 =  0.1
Z2 = -0.1
X3 =  0.1
Z3 =  0.1
X4 = -0.1
Z4 =  0.1


#############  NUMBER AND TYPE OF ELEMENT ALONG BEAM AXIS(Y) ############

n_elem = 1                                                # No of elements
per_elem = 2                                              # Type of the element
n_nodes  = (per_elem-1)*n_elem  + 1                       # Total number of nodes 
Fixed_point = 0                                           # Coordinates of the beam
Free_point  = L


# _____________Mesh generation___________#
coordinate = np.linspace(Fixed_point,Free_point,n_elem+1)




#################  ALONG THE BEAM AXIS(Y)  ##################

xi = np.array([0.57735,-0.57735])                              # Gauss points
W_Length   = 1                                                 # Gauss Weights
Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])                 # Shape functions of a linear(B2) element
N_Der_xi   = np.array([-1/2,1/2])                              # Derivative of the shape function (N,xi)  




#################  ALONG BEAM CROSS SECTION(X,Z) ################


alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])           # Gauss points 
beta  = np.array([0.57735,-0.57735,0.57735,-0.57735])        
W_Cs  = 1                                                       # weight for gauss quadrature in the cross section

#Lagrange polynomials
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])
n_cross_nodes = len(Lag_poly)                                             # No of lagrange nodes per node
DOF = 3                                                                   # Degree of freedom of each lagrange node


# Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         # Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     # with respect to alpha and beta

X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4

# Determinant of Jacobian matrix of the cross section
J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)                      
J_Cs = np.unique(J_Cs)


######################################   STIFFNESS MATRIX COMPUTATION  ##########################################


####  Size of the global stiffness matrix computed using no of nodes and no of cross nodes on each node and DOF  #####
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))    
for l in range(n_elem):
    
    J_Length = N_Der_xi@np.array([[coordinate[l]],            # Jacobian of each element along beam axis
                                 [coordinate[l+1]]])
    

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
                    
                    
                    #Derivative of F wrt to x and z for tau
                    F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
                    F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))

                    #Derivative of F wrt to x and z for s
                    F_s_x = 1/J_Cs*((Z_beta*alpha_der[s])-(Z_alpha*beta_der[s]))
                    F_s_z = 1/J_Cs*((-X_alpha*alpha_der[s])+(X_beta*beta_der[s]))
                    
                    ############  Fundamental nucleus of the stiffness matrix   ############
                    K_xx =  C_22*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                    K_xy =  C_23*np.sum(W_Cs*Lag_poly[tau]*F_s_x*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*F_tau_x*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_xz =  C_12*np.sum(W_Cs*F_tau_z*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_x*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length)     
                    K_yx =  C_44*np.sum(W_Cs*Lag_poly[tau]*F_s_x*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_23*np.sum(W_Cs*F_tau_x*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_yy =  C_55*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_44*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_33*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length) 
                    K_yz =  C_55*np.sum(W_Cs*Lag_poly[tau]*F_s_z*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_13*np.sum(W_Cs*F_tau_z*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)
                    K_zx =  C_12*np.sum(W_Cs*F_tau_x*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_z*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) 
                    K_zy =  C_13*np.sum(W_Cs*Lag_poly[tau]*F_s_z*J_Cs)*np.sum(W_Length*N_Der[i]*Shape_func[j]*J_Length) + C_55*np.sum(W_Cs*F_tau_z*Lag_poly[s]*J_Cs)*np.sum(W_Length*Shape_func[i]*N_Der[j]*J_Length)  
                    K_zz =  C_11*np.sum(W_Cs*F_tau_z*F_s_z*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_66*np.sum(W_Cs*F_tau_x*F_s_x*J_Cs)*np.sum(W_Length*Shape_func[i]*Shape_func[j]*J_Length) + C_55*np.sum(W_Cs*Lag_poly[tau]*Lag_poly[s]*J_Cs)*np.sum(W_Length*N_Der[i]*N_Der[j]*J_Length)
                    F_Nu =  np.array([[K_xx,K_xy,K_xz],[K_yx,K_yy,K_yz],[K_zx,K_zy,K_zz]])
                    
                    
                    if (i==j==0) and (tau == s) and (l==0):
                        np.fill_diagonal(F_Nu,30e12)
                    
                    Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                    
            
                    
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
        
print("The 15*3 component of the Element stiffness matrix is",Elemental_stiffness_matrix[15,3])    
print("")


#######################   ANALYTICAL SOLUTION FROM THE LITERATURE ####################

K_2122 = ((b*L*C_22)/(18*a)) + ((a*L*C_66)/(18*b)) - ((4*a*b*C_44)/(9*L))

print("The Analytical Solution for the above component is",K_2122)


