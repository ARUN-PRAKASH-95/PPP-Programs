                                                               Test Case 1

Aim:
    
    To check whether the implementation of stiffness matrix is correct or not. It is done by comparing one of the components 
of the stiffness matrix (15*3) with the analytical solution given in the literature. Equation (14) in the report.


Parameters:
   
   Component of the matrix to be tested   -  15*3
   Length of the beam (L)                 -  2[m]
   Type of Cross-section                  -  SQUARE
   Side length  (a = b)                   -  0.2[m]
   Youngs modulus (E)                     -  75e9 [Pa]
   Poisson's ratio                        -  0.33
   Type of beam element                   -  Linear (B2)
   Expanison function                     -  Lagrange (L4) polynomial
   Number of beam elements                -  1


Expected Output:
        
        The value of the 15*3 component of the stiffness matrix obtained using implemented code must be closer to the value
obtained using the analytical solution. Equation (14) in the report.


Obtained Result:
    
       The values of the 15*3 component of the stiffness matrix obtained using both code and analytical solution is presented below.

       Analytical solution   -   1.544*10^10
       Implemented code      -   1.522*10^10

The values obtained are closer to each other, which proves that the implementation of stiffness matrix is working well.


Command to run the program:

    The program for test case 1 can be run by simply running the command "python3 Stiffness_comp_check.py" in the terminal. It prints 
both the analytical solution and the value from the implemented code.
      
