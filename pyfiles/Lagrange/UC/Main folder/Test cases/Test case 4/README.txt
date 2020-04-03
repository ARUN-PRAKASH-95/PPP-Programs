                                                                  Test Case 4


Aim:
 
     To compare the maximum and minimum limit of the axial strain (Epsilon_yy) at fixed end cross-section computed using different type of elements and the 
the reference solution. Equation (17) in the report.


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

      The program plots the comparison of axial strain (Epsilon_yy) vs Z at X = 0, Y = 0 (fixed end) computed using the implemented model for 
different type of elements and the reference solution. 40 elements along beam axis is chosen for comparison. 


Obtained result:

      The values of the maximum axial strain obtained using different type of elements and reference equation is given below.

      Reference soln   -    1.000*10^-6 
      B2               -    1.333*10^-5 
      B3               -    1.333*10^-6 
      B4               -    0.448*10^-5 

The implemented model using B2 and B3 element gives the values closer to the reference solution. The value obtained using B4 element
does not match with the reference solution because the L4 expansion across the cross-section might not work well with the B4 element.


Command to run the program:

    The program for test case 4 can be run by simply running the command "python3 Epsilon_yy.py" in the terminal. This programs plots 
the plots the comparison of maximum and minimum limit of the axial strain computed using different type of elements and saves the plot in the name "Strain_limit.png".
          
