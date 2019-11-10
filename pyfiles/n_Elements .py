import numpy as np
import matplotlib.pyplot as pyplot

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

#Along the beam axis(Y)
n_elem = 1                                  # No of elements
n  = 2                                      # No of nodes 
Epsilon = np.array([0.57735,-0.57735])      # Gauss points
Shape_func = np.array([1/2*(1-Epsilon),1/2*(1+Epsilon)])
N_Der      = np.array([-1/2,1/2])           #Derivative of the shape function
J_Length   =  1                             #Jacobian for the length of the beam
W_Length   =  1                             #Weight for the gauss quadrature

#Along the Beam cross section (X,Z)
#Lagrange polynomials
alpha = np.array([-0.57735,0.57735,0.57735,-0.57735])     # Gauss points 
beta  = np.array([-0.57735,-0.57735,0.57735,0.57735])
W_Cs  = 1                                                 #weight for gauss quadrature in the cross section
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])
L_poly = 4

#Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         #Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     #with respect to alpha and beta

X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4


J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)   #Determinant of Jacobian matrix of the cross section


Elemental_stiffness_matrix = np.zeros((n*L_poly*3,n*L_poly*3))
sep = int((n*L_poly*3)/n)        #Seperation point for stacking element stiffness matrix                  

for i in range(n):
    for j in range(n):
        #Fundamental nucleus of the stiffness matrix K_tsij using two point gauss quadrature
        Nodal_stiffness_matrix = np.zeros((L_poly*3,L_poly*3))
        for tau_en,tau in enumerate(range(L_poly)):
            for s_en,s in enumerate(range(L_poly)):
                
                #Fundamental nucleus of the stiffness matrix
                #Derivative of F wrt to x and z for tau
                F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
                F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))

                #Derivative of F wrt to x and z for s
                F_s_x = 1/J_Cs*((Z_beta*alpha_der[s])-(Z_alpha*beta_der[s]))
                F_s_z = 1/J_Cs*((-X_alpha*alpha_der[s])+(X_beta*beta_der[s]))
                
                
                K_xx =  C_22*(W_Cs*np.sum(F_tau_x*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_66*(W_Cs*np.sum(F_tau_z*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_44*(W_Cs*np.sum(Lag_poly[tau]*Lag_poly[s]*J_Cs)*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length))
                K_xy =  C_23*(W_Cs*np.sum(Lag_poly[tau]*F_s_x*J_Cs)*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length)) + C_44*(W_Cs*np.sum(F_tau_x*Lag_poly[s]*J_Cs)*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length))
                K_xz =  C_12*(W_Cs*np.sum(F_tau_z*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_66*(W_Cs*np.sum(F_tau_x*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length))     
                K_yx =  C_44*(W_Cs*np.sum(Lag_poly[tau]*F_s_x*J_Cs)*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length)) + C_23*(W_Cs*np.sum(F_tau_x*Lag_poly[s]*J_Cs)*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length))
                K_yy =  C_55*(W_Cs*np.sum(F_tau_z*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_44*(W_Cs*np.sum(F_tau_x*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_33*(W_Cs*np.sum(Lag_poly[tau]*Lag_poly[s]*J_Cs)*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length)) 
                K_yz =  C_55*(W_Cs*np.sum(Lag_poly[tau]*F_s_z*J_Cs)*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length)) + C_13*(W_Cs*np.sum(F_tau_z*Lag_poly[s]*J_Cs)*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length))
                K_zx =  C_12*(W_Cs*np.sum(F_tau_x*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_66*(W_Cs*np.sum(F_tau_z*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) 
                K_zy =  C_13*(W_Cs*np.sum(Lag_poly[tau]*F_s_z*J_Cs)*W_Length*np.sum(N_Der[i]*Shape_func[j]*J_Length)) + C_55*(W_Cs*np.sum(F_tau_z*Lag_poly[s]*J_Cs)*W_Length*np.sum(Shape_func[i]*N_Der[j]*J_Length))  
                K_zz =  C_11*(W_Cs*np.sum(F_tau_z*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_66*(W_Cs*np.sum(F_tau_x*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_55*(W_Cs*np.sum(Lag_poly[tau]*Lag_poly[s]*J_Cs)*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length))
                F_Nu = np.array([[K_xx,K_xy,K_xz],[K_yx,K_yy,K_yz],[K_zx,K_zy,K_zz]])
                if (i==j==0) and (tau == s):
                    np.fill_diagonal(F_Nu,30e12)
                Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                
        #print(Nodal_stiffness_matrix)
                
        Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
print(Elemental_stiffness_matrix[15,3])
print("Stiffness matrix ----------------------------------------")
print(Elemental_stiffness_matrix)                

Load_vector = np.zeros((n*L_poly*3,1))
Load_vector[14] = -12.5
Load_vector[17] = -12.5
Load_vector[20] = -12.5
Load_vector[23] = -12.5
print("Load vector ----------------------------------------------")
print(Load_vector)

A = np.linalg.inv(Elemental_stiffness_matrix[12:,12:])
C= A@Load_vector[12:]
print("Displacement----------------------------------------------")
print(C)
