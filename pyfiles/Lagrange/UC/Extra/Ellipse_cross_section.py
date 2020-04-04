import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp  
from sympy import *
from sympy.solvers.solveset import linsolve
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm



#PARAMETERS
sma = 0.4          #Semi major axis for the ellipse
smi = 0.2          #Semi minor axis for the ellipse
L = 4              #[m] Length of the beam

X1 = -smi
Z1 = -smi
X2 =  smi
Z2 = -smi
X3 =  smi
Z3 =  smi
X4 = -smi
Z4 =  smi

# X1 = -sma
# Z1 = -smi
# X2 =  sma
# Z2 = -smi
# X3 =  sma
# Z3 =  smi
# X4 = -sma
# Z4 =  smi



#_______________________________________MATERIAL PARAMETERS___________________________________#
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




#______________________________________ELEMENT TYPE_________________________________#

print(" ")
print(" ")
print("Choose the type of element: Linear type (2), Quadratic type (3), Cubic type(4) as input")
per_elem    = int(input("Enter the type of element: "))        # Type of the element
n_elem      = int(input("Enter the number of elements: "))     # No of elements
n_nodes     = (per_elem-1)*n_elem  + 1                         # Total number of nodes 
Fixed_point = 0                                                # Coordinates of the beam
Free_point  = L



# _________________________________________MESH GENERATION ALONG BEAM AXIS(Y)_________________________________________________

coordinate = np.linspace(Fixed_point,Free_point,n_elem+1)
# print(coordinate)
# mrf = 2
# q=mrf**(1/(n_elem-1))
# l=(Fixed_point-Free_point)*(1-q)/(1-mrf*q)
# rnode=Free_point
# c=np.array([Free_point])
# for i in range(n_elem):
#     rnode=rnode+l
#     c=np.append(c,rnode)
#     l=l*q
# coordinate = np.flip(c)





#________________________________SHAPE FUNCTIONS AND GAUSS QUADRATURE FOR BEAM ELEMENT______________________________________#

if (per_elem==2):
    
    #Along the beam axis(Y)
    xi         = 0                                                 # Gauss points
    W_Length   = 2
    Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])                 # Shape functions of a linear element
    N_Der_xi   = np.array([-1/2,1/2])                              # Derivative of the shape function (N,xi)  

elif (per_elem==3):
    
    #Along the beam axis(Y)
    xi = np.array([0.339981,-0.339981,0.861136,0.861136])                                # Gauss points
    W_Length   = np.array([0.652145,0.652145,0.347855,0.347855])                         # Gauss weights
    Shape_func = np.array([1/2*(xi**2-xi),1-xi**2,1/2*(xi**2+xi)])                       # Shape functions
    N_Der_xi = np.array([sp.Symbol('xi')-1/2,-2*sp.Symbol('xi'),sp.Symbol('xi')+1/2])    # Derivative of the shape function 
    N_Der_xi_m = np.array([-1/2,1/2])                                                    # Taking just numerical values from shape function for easily computing jacobian

elif (per_elem==4):
    
    #Along the beam axis(Y)
    xi = np.array([0,0.5384693101056831,-0.5384693101056831,0.9061798459386640,-0.9061798459386640])                                                                     # Gauss points
    W_Length   =   np.array([0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891])           
    Shape_func = np.array([-9/16*(xi+1/3)*(xi-1/3)*(xi-1), 27/16*(xi+1)*(xi-1/3)*(xi-1),-27/16*(xi+1)*(xi+1/3)*(xi-1),9/16*(xi+1/3)*(xi-1/3)*(xi+1)])                       # Shape functions
    N_Der_xi = N_Der_xi = np.array([-1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') + 0.0625,5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') - 1.6875,-5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') + 1.6875,1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') - 0.0625])    # Derivative of the shape function (N,xi)
    N_Der_xi_m = np.array([0.0625,- 1.6875,1.6875,-0.0625]) 





#_______________________________SHAPE FUNCTIONS AND GAUSS QUADRATURE FOR CROSS SECTIONAL ELEMENT________________________________#



#Along the Beam cross section (X,Z)
#Lagrange polynomials
alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])           # Gauss points 
beta  = np.array([0.57735,-0.57735,0.57735,-0.57735]) 
W_Cs  = 1                                                                             # WeightS for gauss quadrature in the cross section
Lag_poly = np.array([1/4*(1-alpha)*(1-beta),1/4*(1+alpha)*(1-beta),1/4*(1+alpha)*(1+beta),1/4*(1-alpha)*(1+beta)])
n_cross_nodes = len(Lag_poly)                                                         # No of lagrange nodes per node
DOF = 3                                                                               # Degree of freedom of each lagrange node

#Lagrange Derivatives
alpha_der = np.array([-1/4*(1-beta),1/4*(1-beta),1/4*(1+beta),-1/4*(1+beta)])         # Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-alpha),-1/4*(1+alpha),1/4*(1+alpha),1/4*(1-alpha)])     # with respect to alpha and beta

X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4


J_Cs = (Z_beta*X_alpha - Z_alpha*X_beta)                                               # Determinant of Jacobian matrix of the cross section
J_Cs = J_Cs[0]





#__________________________________________STIFFNESS MATRIX COMPUTATION____________________________________________________#


#Size of the global stiffness matrix computed using no of nodes and no of cross nodes on each node and DOF
Global_stiffness_matrix = np.zeros((n_nodes*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))    
for l in tqdm(range(n_elem)):

    if (per_elem==2):
        J_Length = N_Der_xi@np.array([[coordinate[l]],            # Jacobian of each element along beam axis
                                     [coordinate[l+1]]])
    
        # Derivative of the shape functions with respect to physical coordinates (N,y)
        N_Der    = np.array([-1/2*(1/J_Length),1/2*(1/J_Length)]) 


    elif (per_elem==3):
        X_coor     = np.array([[coordinate[l]],
                               [coordinate[l+1]]])                                                        
        J_Length   = N_Der_xi_m@X_coor                             # Jacobian of each element along beam axis
        
        # Derivative of the shape function wrt to physical coordinates(N,y) 
        N_Der      = np.array([(xi-1/2)*(1/J_Length),-2*xi*(1/J_Length),(xi+1/2)*(1/J_Length)]) 

    elif (per_elem==4):
        mid = (coordinate[l+1]+coordinate[l])/2
        mid_length = (coordinate[l+1]-coordinate[l])/2 
        X_coor = np.array([[coordinate[l]],
                           [mid-(mid_length/3)], 
                           [mid+(mid_length/3)],
                           [coordinate[l+1]]])                                                      

                  
        J_Length   = N_Der_xi_m@X_coor                                            # Jacobian for the length of the beam
        N_Der      = np.array([(-1.6875*xi**2 + 1.125*xi + 0.0625)*(1/J_Length),(5.0625*xi**2 - 1.125*xi - 1.6875)*(1/J_Length),(-5.0625*xi**2 - 1.125*xi + 1.6875)*(1/J_Length),(1.6875*xi**2 + 1.125*xi - 0.0625)*(1/J_Length)])        # Derivative of the shape function wrt to physical coordinates(N,y)

               
     
    

    # Element stiffness matrix created using no of nodes per element and cross node and DOF  
    Elemental_stiffness_matrix = np.zeros((per_elem*n_cross_nodes*DOF,per_elem*n_cross_nodes*DOF)) 
    sep = int((per_elem*n_cross_nodes*DOF)/per_elem)                             # Seperation point for stacking element stiffness matrix                  

    for i in range(per_elem):
        for j in range(per_elem):
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
                    
                    #Fundamental nucleus of the stiffness matrix
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
                        np.fill_diagonal(F_Nu,30e12)
                    
                    Nodal_stiffness_matrix[3*s:3*(s+1) , 3*tau:3*(tau+1)]  = F_Nu
                    
            
                    
            Elemental_stiffness_matrix[sep*j:sep*(j+1) , sep*i:sep*(i+1)] = Nodal_stiffness_matrix
        
    if (per_elem==2):
        
        #Assignment matix for arranging global stiffness matrix
        A_fac = 12
        Ae = np.zeros((per_elem*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
        np.fill_diagonal( Ae[0:A_fac*2 , A_fac*l:A_fac*(l+2)] ,1)
        AeT = np.transpose(Ae)
        K = AeT@Elemental_stiffness_matrix@Ae
        Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

    elif (per_elem==3):

        #Assignment matix for arranging global stiffness matrix
        A_fac = 12
        Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
        np.fill_diagonal( Ae[A_fac*0:A_fac*3 , A_fac*2*l:A_fac*(3+(2*l))] , 1 )
        AeT = np.transpose(Ae)
        K = AeT@Elemental_stiffness_matrix@Ae
        Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

    elif (per_elem==4):

        #Assignment matix for arranging global stiffness matrix
        A_fac = 12
        Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))
        np.fill_diagonal( Ae[A_fac*0:A_fac*4 , A_fac*3*l:A_fac*(4+(3*l))] , 1 )
        AeT = np.transpose(Ae)
        K = AeT@Elemental_stiffness_matrix@Ae
        Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)



Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
Load_vector[n_nodes*n_cross_nodes*DOF-10] = -25
Load_vector[n_nodes*n_cross_nodes*DOF-7]  = -25
Load_vector[n_nodes*n_cross_nodes*DOF-4]  = -25
Load_vector[n_nodes*n_cross_nodes*DOF-1]  = -25


Displacement = np.linalg.solve(Global_stiffness_matrix,Load_vector)
print(Displacement)

#____________________________________________POST PROCESSING PHASE______________________________________________________#

#To extract the displacement of our interest 

#X displacements of all the lagrange nodes
Displacement[n_nodes*n_cross_nodes*DOF-11] =  -Displacement[n_nodes*n_cross_nodes*DOF-11] 
Displacement[n_nodes*n_cross_nodes*DOF-5]  =  -Displacement[n_nodes*n_cross_nodes*DOF-5]
Displacement[13] =  -Displacement[13] 
Displacement[19] =  -Displacement[19]

X_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    X_disp = np.append(X_disp,Displacement[3*(k+1)-3])

# Req_X_disp = X_disp[-4::]                       #Displacement of the lagrange nodes at end cross section
Req_X_disp = X_disp[4:8]


#Y displacements of all the lagrange nodes
Y_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    Y_disp = np.append(Y_disp,Displacement[3*(k+1)-2])

# Req_Y_disp = Y_disp[-4::]
Req_Y_disp = Y_disp[4:8]


#Z displacements of all the lagrange nodes
Z_disp = np.array([])
for k in range(n_nodes*n_cross_nodes):
    Z_disp = np.append(Z_disp,Displacement[3*(k+1)-1])

# Req_Z_disp = Z_disp[-4::]
Req_Z_disp = Z_disp[4:8]

#Z_displacement of the center point of all cross section
Z_disp_cen = np.array([])
for k in range(n_nodes):
    Z_disp_cen = np.append(Z_disp_cen,Z_disp[4*(k+1)-3])

alpha,beta = symbols('alpha,beta')
F1 = 1/4*(1-alpha)*(1-beta)
F2 = 1/4*(1+alpha)*(1-beta)
F3 = 1/4*(1+alpha)*(1+beta)
F4 = 1/4*(1-alpha)*(1+beta)



#Creating a mesh grid of major axis to create coordinates inside an ellipse
a = np.linspace(-sma,sma,100)
b = np.linspace(-sma,sma,100)

aa,bb = np.meshgrid(a,b)    #Mesh grid 
ell = (aa**2/sma**2) + (bb**2/smi**2)   #Ellipse equation to get points inside the ellipse


rows,col = ell.shape
ell_x = np.array([])
ell_y = np.array([])
for i in tqdm(range(rows)):
    for j in range(col):
        if(ell[i,j]<=1):
            ell_x = np.append(ell_x,aa[i,j])
            ell_y = np.append(ell_y,bb[i,j])


# X1 = -sma
# Z1 = -sma
# X2 =  sma
# Z2 = -sma
# X3 =  sma
# Z3 =  sma
# X4 = -sma
# Z4 =  sma


# X1 = -sma
# Z1 = -smi
# X2 =  sma
# Z2 = -smi
# X3 =  sma
# Z3 =  smi
# X4 = -sma
# Z4 =  smi


X1 = -smi
Z1 = -smi
X2 =  smi
Z2 = -smi
X3 =  smi
Z3 =  smi
X4 = -smi
Z4 =  smi

########### Sympy equation to find natural coordinates of the physical domain #############
coor = np.array([])
for i in tqdm(range(len(ell_x))):
    eq1 =  F1*X1 + F2 * X2 + F3 * X3 + F4 * X4 - ell_x[i]
    eq2 =  F1*Z1 + F2 * Z2 + F3 * Z3 + F4 * Z4 - ell_y[i]
    a = solve([eq1, eq2], (alpha,beta))
    coor=np.append(coor,a)



#Natural coordinates of the points in the physical domain
X_nat = np.array([])
Y_nat = np.array([])

for i in range(len(coor)):
    x_nat = coor[i][alpha]
    y_nat = coor[i][beta]
    X_nat = np.append(X_nat,x_nat)
    Y_nat = np.append(Y_nat,y_nat)

Lag_poly = np.array([1/4*(1-X_nat)*(1-Y_nat),1/4*(1+X_nat)*(1-Y_nat),1/4*(1+X_nat)*(1+Y_nat),1/4*(1-X_nat)*(1+Y_nat)])


########################  REQUIRED DISPLACEMENTS (X,Y AND Z) OF THE MESH GRID (CROSS SECTION) #########################
X_Req = Lag_poly[0]*Req_X_disp[0] + Lag_poly[1]*Req_X_disp[1] + Lag_poly[2]*Req_X_disp[2]  + Lag_poly[3]*Req_X_disp[3]
Y_Req = Lag_poly[0]*Req_Y_disp[0] + Lag_poly[1]*Req_Y_disp[1] + Lag_poly[2]*Req_Y_disp[2]  + Lag_poly[3]*Req_Y_disp[3]
Z_Req = Lag_poly[0]*Req_Z_disp[0] + Lag_poly[1]*Req_Z_disp[1] + Lag_poly[2]*Req_Z_disp[2]  + Lag_poly[3]*Req_Z_disp[3]
# print(Y_Req)




############################   AXIAL STRAINS (Epsilon_xx, Epsilon_yy and Epsilon_zz)  #################################

#____________________Strains in Y axis (Epsilon_yy)____________________#
Epsilon_yy =  (Lag_poly[0]*1/2*(1/J_Length)*Req_Y_disp[0] + Lag_poly[1]*1/2*(1/J_Length)*Req_Y_disp[1] + Lag_poly[2]*1/2*(1/J_Length)*Req_Y_disp[2] + Lag_poly[3]*1/2*(1/J_Length)*Req_Y_disp[3])*2
# print("Epsilon_yy",E*Epsilon_yy)


#_____________Strains in X and Z axis (Epsilon_xx and Epsilon_zz)_____________#
alpha_der = np.array([-1/4*(1-Y_nat),1/4*(1-Y_nat),1/4*(1+Y_nat),-1/4*(1+Y_nat)])         # Derivatives of the lagrange polynomials
beta_der  = np.array([-1/4*(1-X_nat),-1/4*(1+X_nat),1/4*(1+X_nat),1/4*(1-X_nat)])         # with respect to alpha and beta

#_____________Derivative of X and z with respect to alpha and beta_____________#
X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4

Epsilon_xx = (1/J_Cs)*((Z_beta*alpha_der[0])-(Z_alpha*beta_der[0]))*Req_X_disp[0] + (1/J_Cs)*((Z_beta*alpha_der[1])-(Z_alpha*beta_der[1]))*Req_X_disp[1] + (1/J_Cs)*((Z_beta*alpha_der[2])-(Z_alpha*beta_der[2]))*Req_X_disp[2] + (1/J_Cs)*((Z_beta*alpha_der[3])-(Z_alpha*beta_der[3]))*Req_X_disp[3] 
Epsilon_zz = 1/J_Cs*((-X_alpha*alpha_der[0])+(X_beta*beta_der[0]))*Req_Z_disp[0] + 1/J_Cs*((-X_alpha*alpha_der[1])+(X_beta*beta_der[1]))*Req_Z_disp[1] + 1/J_Cs*((-X_alpha*alpha_der[2])+(X_beta*beta_der[2]))*Req_Z_disp[2] + 1/J_Cs*((-X_alpha*alpha_der[3])+(X_beta*beta_der[3]))*Req_Z_disp[3] 

#_________________________________________________________________________________________________________________________#



N_Epsilon_yy = np.array([])
for i in Epsilon_yy:
    N_Epsilon_yy = np.append(N_Epsilon_yy,round(i*10**9,5))

N_Y_Req = np.array([])
for i in Y_Req:
    N_Y_Req = np.append(N_Y_Req,round(i*10**8,5))



#___________Z_displacement of the center of all the nodes________#
# fig,ax = plt.subplots()
# co=np.linspace(Fixed_point,Free_point,n_nodes)
# ax.plot(co,Z_disp_cen*10**7,marker='o')
# ax.set_title("Z_displacement of the central point of all cross sections")
# ax.set_xlabel('Coordinates of the beam along beam axis[Y]')
# ax.set_ylabel('$u_{z}[10^{-7}m]$')
# plt.savefig('Z_Displacement_ell.png')




#___________Plots Y displacment of the end cross section(shows bending behaviour)________#    
fig,ax = plt.subplots()
ax = plt.axes(projection='3d')
ax.scatter(ell_x,ell_y,N_Y_Req)
ax.set(xlabel = "X", ylabel = "Z", zlabel="$u_{y}[10^{-8}m]$", title='(B'+ str(per_elem)+')element along beam axis(Y)')
plt.savefig('B'+str(per_elem)+'_Y_Displacement_ell.png')





#____________Plots the axial strain(Epsilon_yy) of the end cross section_________#
fig,ax = plt.subplots()
cb=ax.scatter(ell_x,ell_y,c=N_Epsilon_yy)
ax.set(xlabel='X[m]',ylabel='Z[m]',title='Strain ($\epsilon_{yy}$) of the end cross section')
plt.colorbar(cb,label='$\epsilon_{yy}[10^{-9}]$')
plt.savefig('Strain_ell.png')




#____________Plots the axial strain(Sigma_yy) of the end cross section_________#
fig,ax = plt.subplots()
cb=ax.scatter(ell_x,ell_y,c=E*Epsilon_yy*10**-6,cmap='RdBu')
ax.set(xlabel='X[m]',ylabel='Z[m]',title='Stress ($\sigma_{yy}$) of the end cross section')
plt.colorbar(cb,label='$\sigma_{yy}[MPa]$')
plt.savefig('Stress_ell.png')
