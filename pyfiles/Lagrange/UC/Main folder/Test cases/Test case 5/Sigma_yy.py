                                                ########## TEST CASE 5  ########


# OBJECTIVE
# To compare the maximum and minimum limit of the Axial stress (Sigma_yy) with the reference solution #




import numpy as np
import matplotlib.pyplot as plt  
import sympy as sp  
from sympy import *
from sympy.solvers.solveset import linsolve
from mpl_toolkits.mplot3d import Axes3D



############################ MATERIAL AND GEOMETRY PARAMETERS  #############################

a = 0.2            #[m] Square cross section
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

per_element = np.array([2,3,4])    # Type of the element
n_elem = np.array([40])            # No of elements
epsilon_yy = np.array([])          # Array for storing the z displacement of central point of tip cross section



for per_elem in per_element: 

    for i in n_elem:
        n_nodes  = (per_elem-1)*i  + 1                       # Total number of nodes 
        Fixed_point = 0                                      # Coordinates of the beam
        Free_point  = L


        #############################      MESH GENERATION ALONG BEAM AXIS(Y)      #######################################


        coordinate = np.linspace(Fixed_point,Free_point, i+1)



        #________________________________SHAPE FUNCTIONS AND GAUSS QUADRATURE FOR BEAM ELEMENT______________________________________#

        if (per_elem==2):
            
            # For Linear(B2) Element
            xi         = 0                                                 # Gauss points
            W_Length   = 2                                                 # Gauss Weights
            Shape_func = np.array([1/2*(1-xi),1/2*(1+xi)])                 # Shape functions of a linear element
            N_Der_xi   = np.array([-1/2,1/2])                              # Derivative of the shape function (N,xi)  

        
        elif (per_elem==3):
            
            # For Quadratic(B3) Element
            xi = np.array([0.339981,-0.339981,0.861136,0.861136])                                # Gauss points
            W_Length   = np.array([0.652145,0.652145,0.347855,0.347855])  
            Shape_func = np.array([1/2*(xi**2-xi),1-xi**2,1/2*(xi**2+xi)])                       # Shape functions
            N_Der_xi = np.array([sp.Symbol('xi')-1/2,-2*sp.Symbol('xi'),sp.Symbol('xi')+1/2])    # Derivative of the shape function 
            N_Der_xi_m = np.array([-1/2,1/2])                                                    # Taking just numerical values from shape function for easily computing jacobian

        
        elif (per_elem==4):
            
            # For Cubic(B4) Element
            xi = np.array([0,0.5384693101056831,-0.5384693101056831,0.9061798459386640,-0.9061798459386640])                                                                     # Gauss points
            W_Length   =   np.array([0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891])           
            Shape_func = np.array([-9/16*(xi+1/3)*(xi-1/3)*(xi-1), 27/16*(xi+1)*(xi-1/3)*(xi-1),-27/16*(xi+1)*(xi+1/3)*(xi-1),9/16*(xi+1/3)*(xi-1/3)*(xi+1)])                       # Shape functions
            N_Der_xi = N_Der_xi = np.array([-1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') + 0.0625,5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') - 1.6875,-5.0625*sp.Symbol('xi')**2 - 1.125*sp.Symbol('xi') + 1.6875,1.6875*sp.Symbol('xi')**2 + 1.125*sp.Symbol('xi') - 0.0625])    # Derivative of the shape function (N,xi)
            N_Der_xi_m = np.array([0.0625,- 1.6875,1.6875,-0.0625]) 





        #_______________________________SHAPE FUNCTIONS AND GAUSS QUADRATURE FOR CROSS SECTIONAL ELEMENT________________________________#



        #################  ALONG BEAM CROSS SECTION(X,Z) ################
        
        #Lagrange polynomials
        alpha = np.array([0.57735,0.57735,-0.57735,-0.57735])                                 # Gauss points 
        beta  = np.array([0.57735,-0.57735,0.57735,-0.57735]) 
        W_Cs  = 1                                                                             # Weight for gauss quadrature in the cross section
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
        for l in range(i):
            
            
            # For Linear(B2) Element
            if (per_elem==2):
                J_Length = N_Der_xi@np.array([[coordinate[l]],            # Jacobian of each element along beam axis
                                            [coordinate[l+1]]])
            
                ### Derivative of the shape functions with respect to physical coordinates (N,y) ###
                N_Der    = np.array([-1/2*(1/J_Length),1/2*(1/J_Length)]) 

            
            
            # For Quadratic(B3) Element
            elif (per_elem==3):
                X_coor     = np.array([[coordinate[l]],
                                    [coordinate[l+1]]])                                                        
                J_Length   = N_Der_xi_m@X_coor                             # Jacobian of each element along beam axis
                
                ###### Derivative of the shape function wrt to physical coordinates(N,y) #####
                N_Der      = np.array([(xi-1/2)*(1/J_Length),-2*xi*(1/J_Length),(xi+1/2)*(1/J_Length)]) 

            
            # For Cubic(B4) element
            elif (per_elem==4):
                mid = (coordinate[l+1]+coordinate[l])/2
                mid_length = (coordinate[l+1]-coordinate[l])/2 
                X_coor = np.array([[coordinate[l]],                        
                                [mid-(mid_length/3)], 
                                [mid+(mid_length/3)],
                                [coordinate[l+1]]])                                                      

                        
                J_Length   = N_Der_xi_m@X_coor                             # Jacobian of each element along beam axis
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
                            
                            
                            #############  Fundamental nucleus of the stiffness matrix  #############
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
                
                #Assignment matix for arranging global stiffness matrix for B2 element
                A_fac = 12
                Ae = np.zeros((per_elem*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
                np.fill_diagonal( Ae[0:A_fac*2 , A_fac*l:A_fac*(l+2)] ,1)
                AeT = np.transpose(Ae)
                K = AeT@Elemental_stiffness_matrix@Ae
                Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

            
            elif (per_elem==3):

                #Assignment matix for arranging global stiffness matrix for B3 element
                A_fac = 12
                Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))       
                np.fill_diagonal( Ae[A_fac*0:A_fac*3 , A_fac*2*l:A_fac*(3+(2*l))] , 1 )
                AeT = np.transpose(Ae)
                K = AeT@Elemental_stiffness_matrix@Ae
                Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)

            
            elif (per_elem==4):

                #Assignment matix for arranging global stiffness matrix for B4 element
                A_fac = 12
                Ae = np.zeros((len(Shape_func)*n_cross_nodes*DOF,n_nodes*n_cross_nodes*DOF))
                np.fill_diagonal( Ae[A_fac*0:A_fac*4 , A_fac*3*l:A_fac*(4+(3*l))] , 1 )
                AeT = np.transpose(Ae)
                K = AeT@Elemental_stiffness_matrix@Ae
                Global_stiffness_matrix = np.add(Global_stiffness_matrix,K)




        #____________________________________________LOAD VECTOR____________________________________________________#

        Load_vector = np.zeros((n_nodes*n_cross_nodes*DOF,1))
        Load_vector[n_nodes*n_cross_nodes*DOF-10] = -12.5
        Load_vector[n_nodes*n_cross_nodes*DOF-7]  = -12.5
        Load_vector[n_nodes*n_cross_nodes*DOF-4]  = -12.5
        Load_vector[n_nodes*n_cross_nodes*DOF-1]  = -12.5

        ############## SOLVES FUNDAMENTAL EQUATION OF FEA ##############
        Displacement = np.linalg.solve(Global_stiffness_matrix,Load_vector)
        


        #____________________________________________POST PROCESSING PHASE______________________________________________________#

        #To extract the displacement of our interest 
        Displacement[n_nodes*n_cross_nodes*DOF-11] =  -Displacement[n_nodes*n_cross_nodes*DOF-11] 
        Displacement[n_nodes*n_cross_nodes*DOF-5]  =  -Displacement[n_nodes*n_cross_nodes*DOF-5]
        Displacement[13] =  -Displacement[13] 
        Displacement[19] =  -Displacement[19]
        

        
        #X displacements of all the lagrange nodes
        X_disp = np.array([])
        for k in range(n_nodes*n_cross_nodes):
            X_disp = np.append(X_disp,Displacement[3*(k+1)-3])
        Req_X_disp = X_disp[4:8]     # X Displacements of the lagrange nodes at the cross section close to fixed end

        
        #Y displacements of all the lagrange nodes
        Y_disp = np.array([])
        for k in range(n_nodes*n_cross_nodes):
            Y_disp = np.append(Y_disp,Displacement[3*(k+1)-2])
        Req_Y_disp = Y_disp[4:8]     # Y Displacements of the lagrange nodes at the cross section close to fixed end
        

        #Z displacements of all the lagrange nodes
        Z_disp = np.array([])
        for k in range(n_nodes*n_cross_nodes):
            Z_disp = np.append(Z_disp,Displacement[3*(k+1)-1])
        Req_Z_disp = Z_disp[4:8]      # Z Displacements of the lagrange nodes at the cross section close to fixed end



        
        
        #############   Calculates Axial Strain(Epsilon_yy) at X=0,Y=O and along Z   ###########
        X_nat = np.full((10),0)
        z_nat = np.linspace(-1,1,10)
        Lag_poly = np.array([1/4*(1-X_nat)*(1-z_nat),1/4*(1+X_nat)*(1-z_nat),1/4*(1+X_nat)*(1+z_nat),1/4*(1-X_nat)*(1+z_nat)])
        
        Epsilon_yy =  (Lag_poly[0]*1/2*(1/J_Length)*Req_Y_disp[0] + Lag_poly[1]*1/2*(1/J_Length)*Req_Y_disp[1] + Lag_poly[2]*1/2*(1/J_Length)*Req_Y_disp[2] + Lag_poly[3]*1/2*(1/J_Length)*Req_Y_disp[3])*2 
        epsilon_yy = np.append(epsilon_yy,Epsilon_yy)
        
        

        


################ Exact solution for the maximum and minimum limit of the Axial stress (Sigma_yy) ###########
h = np.linspace(-0.1,0.1,10)
a = 0.2
exact_epsilon_yy = (50*2*h*12)/(E*a**4)    



##################### PLOTS THE AXIAL STRESS(SIGMA_YY) VS Z AT X=0,Y=O (GIVES STRESS LIMIT AT FIXED CROSS SECTION) ####################

fig,ax = plt.subplots()
ax.plot(h,E*epsilon_yy[: 10]*10**-6,marker='o',label='Linear(B2)')           # Plots Axial strain(Sigma_yy) of B2 element
ax.plot(h,E*epsilon_yy[10 : 20]*10**-6,marker='*',label='Quadratic(B3)')     # Plots Axial strain(Sigma_yy) of B3 element
ax.plot(h,E*epsilon_yy[20 : 30]*10**-6,marker='x',label='Cubic(B4)')         # Plots Axial strain(Sigma_yy) of B4 element
ax.plot(h,E*exact_epsilon_yy*10**-6,marker='+',label='Exact')
ax.set(xlabel='Z [m]',ylabel='$\sigma_{yy} [MPa]$',title='Axial stress ($\sigma_{yy}$) vs Z')
ax.legend()
plt.savefig('Stress_limit.png')
