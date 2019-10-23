import numpy as np
import matplotlib.pyplot as pyplot

#PARAMETERS
a = 0.2            #[m] Square cross section
L = 2              #[m] Length of the beam
E = 75e9           #[Pa] Young's Modulus
v = 0.33           # Poissons Ratio
G = E/(2*(1+v))
First  = (E*(1-v))/((1+v)*(1-2*v))
Second = v*E/((1+v)*(1-2*v))
C_22 = First
C_44 = G
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
n  = 1                  # No  of elements
Epsilon = np.array([0.57735,-0.57735])    #Gauss points
Shape_func = np.array([1/2*(1-Epsilon),1/2*(1+Epsilon)])
N_Der      = np.array([-1/2,1/2])       #Derivative of the shape function
J_Length   =  1                         #Jacobian for the length of the beam
W_Length   =  1                         #Weight for the gauss quadrature

#Along the Beam cross section (X,Z)
#Lagrange polynomials
alpha = np.array([-0.57735,0.57735,0.57735,-0.57735])     # Gauss points 
beta  = np.array([-0.57735,-0.57735,0.57735,0.57735])
W_Cs  = 1
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])

#Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         #Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     #with respect to alpha and beta

X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4

J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)   #Determinant of Jacobian matrix of the cross section


#Fundamental nucleus of the stiffness matrix
#Derivative of F wrt to x and z for tau
tau = 1
F_tau_x = 1/J_Cs*((Z_beta*alpha_der[tau])-(Z_alpha*beta_der[tau]))
F_tau_z = 1/J_Cs*((-X_alpha*alpha_der[tau])+(X_beta*beta_der[tau]))

#Derivative of F wrt to x and z for s
s = 1
F_s_x = 1/J_Cs*((Z_beta*alpha_der[s])-(Z_alpha*beta_der[s]))
F_s_z = 1/J_Cs*((-X_alpha*alpha_der[s])+(X_beta*beta_der[s]))


i = 0
j = 1
#
K_xx =  C_22*(W_Cs*np.sum(F_tau_x*F_s_x*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_66*(W_Cs*np.sum(F_tau_z*F_s_z*J_Cs)*W_Length*np.sum(Shape_func[i]*Shape_func[j]*J_Length)) + C_44*(W_Cs*np.sum(Lag_poly[tau]*Lag_poly[s]*J_Cs)*W_Length*np.sum(N_Der[i]*N_Der[j]*J_Length))

