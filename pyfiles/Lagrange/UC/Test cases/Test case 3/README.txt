                                                                  Test Case 3

Aim:
  
       To compare the  Z displacement of the central point of the tip cross-section due to bending load computed by our model with the reference solution given in the literature. 
Equation (16) in the report.



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
   Number of beam elements                -  1, 2, 10, 20, 40



Expected output:
         
        The program plots the comparison of vertical displacements of the tip cross-secction for different type of beam elements 
and the displacement obtained from the reference solution (Equation (16)).  1, 2, 10, 20, 40 number of elements are considered for 
the comparison.



Obtained result:
 
        The values of the maximum vertical displacement obtained using different type of elements and reference equation is 
given below.

           Reference soln   -    -1.333*10^-5 [m]
           B2               -    -0.900*10^-5 [m]
           B3               -    -1.283*10^-5 [m]
           B4               -    -0.908*10^-5 [m]

The implemented model using B3 element as beam element gives value closer to the reference solution. The value obtained using B4 element
does not match with the reference solution because the L4 expansion across the cross-section might not work well with the B4 element.



Command to run the program:

    The program for test case 3 can be run by simply running the command "python3 Z_displacement.py" in the terminal. This programs plots 
the plots the comparison of vertical displacement computed using different type of elements and saves the plot in the name "Z_Displacement.png".
          
