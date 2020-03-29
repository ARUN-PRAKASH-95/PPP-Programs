     # ##################################    NON-AXIAL STRAIN (Epsilon_yz)   ###########################################

        
        # X_nat = np.full((10),-1)
        # z_nat = np.linspace(-1,1,10)
        
        # Lag_poly = np.array([1/4*(1-X_nat)*(1-z_nat),1/4*(1+X_nat)*(1-z_nat),1/4*(1+X_nat)*(1+z_nat),1/4*(1-X_nat)*(1+z_nat)])

        # alpha_der = np.array([-1/4*(1-z_nat),1/4*(1-z_nat),1/4*(1+z_nat),-1/4*(1+z_nat)])         # Derivatives of the lagrange polynomials
        # beta_der  = np.array([-1/4*(1-X_nat),-1/4*(1+X_nat),1/4*(1+X_nat),1/4*(1-X_nat)])         # with respect to alpha and beta
        # X_alpha = alpha_der[0]*X1 + alpha_der[1]*X2 + alpha_der[2]*X3 + alpha_der[3]*X4
        # X_beta  = beta_der[0] *X1 + beta_der[1]*X2  + beta_der[2] *X3 + beta_der[3] *X4
        # Z_alpha = alpha_der[0]*Z1 + alpha_der[1]*Z2 + alpha_der[2]*Z3 + alpha_der[3]*Z4
        # Z_beta  = beta_der[0] *Z1 + beta_der[1]*Z2  + beta_der[2] *Z3 + beta_der[3] *Z4


        # #_____________________ Derivative of y_displacement w.r.t to z (U_y,z)_________________________#
        # U_yz = 1/J_Cs*((-X_alpha*alpha_der[0])+(X_beta*beta_der[0]))*Req_Y_disp[0] + 1/J_Cs*((-X_alpha*alpha_der[1])+(X_beta*beta_der[1]))*Req_Y_disp[1] + 1/J_Cs*((-X_alpha*alpha_der[2])+(X_beta*beta_der[2]))*Req_Y_disp[2] + 1/J_Cs*((-X_alpha*alpha_der[3])+(X_beta*beta_der[3]))*Req_Y_disp[3] 
        # # print("U_yz",U_yz)
        # #_____________________ Derivative of z_displacement w.r.t to z (U_z,y)_________________________#
        # U_zy = Lag_poly[0]*1/2*(1/J_Length)*Req_Z_disp[0] + Lag_poly[1]*1/2*(1/J_Length)*Req_Z_disp[1] + Lag_poly[2]*1/2*(1/J_Length)*Req_Z_disp[2] + Lag_poly[3]*1/2*(1/J_Length)*Req_Z_disp[3] 

        # Epsilon_yz = 1/2*(U_yz + U_zy)
        # epsilon_yz = np.append(epsilon_yz,Epsilon_yz)

        # print("Epsilon_yz",Epsilon_yz)
        # print(Epsilon_yz.shape)
        


##################### PLOTS THE NON-AXIAL STRESS(SIGMA_YZ) VS Z AT X=-b/2,Y=L (GIVES STRESS LIMIT AT TIP CROSS SECTION) ####################

# fig,ax = plt.subplots()
# ax.plot(h,E*epsilon_yz[: 10],marker='o',label='Linear(B2)')
# ax.plot(h,E*epsilon_yz[10 : 20],marker='*',label='Quadratic(B3)')
# ax.plot(h,E*epsilon_yz[20 : 30],marker='x',label='Cubic(B4)')
# ax.plot(h,exact_sigma_yz,marker='+',label='Exact')
# ax.set(xlabel='Z [m]',ylabel='$\sigma_{yy}[pa]$',title='Axial stress ($\sigma_{yy}$) vs Z')
# ax.legend()
# plt.savefig('Stress_limit_yz.png')


# P = 50
# exact_sigma_yz = 4.8077*P/(h**2)              #Non_axial strain(Epsilon_yz)

# print(exact_epsilon_yy)
# print("Epsilon_yy",Epsilon_yy)