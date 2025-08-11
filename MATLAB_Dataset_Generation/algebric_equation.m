function F = algebric_equation(x)

global I_val P_t delZ H2

x;

F = x- (delZ * (1 + (I_val + P_t) / K_function(x)) + H2);