                                                                  Test Case 2


Aim:
   
    To check whether the displacments obtained by the implemented model is correct or not.  It is done by performing simple tension test
and comparing the displacement value with the reference solution. Equation (15) in the report.


Parameters:
   
   Type of Load applied                   -  Tension 
   Magnitude of the load                  -  50[N]
   Length of the beam (L)                 -  2[m]
   Type of Cross-section                  -  SQUARE
   Side length  (a = b)                   -  0.2[m]
   Youngs modulus (E)                     -  75e9 [Pa]
   Poisson's ratio                        -  0.33
   Type of beam element                   -  Linear (B2)
   Expanison function                     -  Lagrange (L4) polynomial
   Number of beam elements                -  20


Expected output:

    The displacement obtained due to tension load in positive y direction of the implemented code should match with the reference
solution. Equation (15) in the report.


Obtained Result:
   
    The Values of the y displacement obtained using both implemented code and reference solution is given below.

    Analytical solution   -   3.33*10^-8
    Implemented code      -   2.96*10^-8

The values obtained are closer to each other, which proves that the load vector derived using the refined 1D beam model is correct.


Command to run the program:

    The program for test case 2 can be run by simply running the command "python3 Tension_test.py" in the terminal. It prints 
both the analytical solution and the value from the implemented code.
