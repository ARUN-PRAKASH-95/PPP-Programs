                                                                  Test Case 4


Aim:
 
     To compare the maximum and minimum limit of the axial stress (Sigma_yy) at fixed end cross-section computed using different type of elements and the 
the reference solution. Equation (18) in the report.


Parameters:
   
   Type of Load applied                   -  Bending
   Magnitude of the load                  -  -50[N]
   Length of the beam (L)                 -  2[m]
   Type of Cross-section                  -  SQUARE
   Side length  (a = b)                   -  0.2[m]
   Youngs modulus (E)                     -  75e9 [Pa]
   Poisson's ratio                        -  0.33
   Type of beam element                   -  Linear (B2), Quadratic (B3) and Cubic (B4)
   Expanison function                     -  Lagrange (L4) polynomial
   Number of beam elements                -  40


Expected output:

      The program plots the comparison of axial stress (Sigma_yy) vs Z at X = 0, Y = 0 (fixed end) computed using the implemented model for 
different type of elements and the reference solution. 40 elements along beam axis is chosen for comparison. 


Obtained result:

      The values of the maximum axial stress obtained using different type of elements and reference equation is given below.

      Reference soln   -    0.075 [MPa]
      B2               -    0.099 [MPa]
      B3               -    0.099 [MPa] 
      B4               -    0.033 [MPa] 

The implemented model using B2 and B3 element gives the values closer to the reference solution. The value obtained using B4 element
does not match with the reference solution because the L4 expansion across the cross-section might not work well with the B4 element.


Command to run the program:

    The program for test case 5 can be run by simply running the command "python3 Sigma_yy.py" in the terminal. This programs plots 
the plots the comparison of maximum and minimum limit of the axial stress computed using different type of elements and saves the plot in the name "Strain_limit.png".
          
