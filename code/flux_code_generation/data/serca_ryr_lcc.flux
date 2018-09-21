################################################################################
# Main file defining fluxes for the code generator.
#
# *Flux* is the main object that describes current. The first argument is 
# its name. Second argument is optional. For fluxes that need to change states
# stochastically, it should be a file name where the corresponding Markov model
# is described. All other arguments are the variables that are generated as
# fields in corresponding structures together with their default values.
#
# The calculations between different *Flux* objects are used to generate
# a function flux. The return value should be stored in a variable d_<name>
# where <name> is the flux name stored in the first argument.
#
# Except varaibles passed to the *Flux* object the following parametrized 
# variables can be used: dt, h, u0, u1, i etc.
# 
################################################################################


# this will create a flux structure with given fields and their default values
# By default variables "params" and "boundary_name" will be created
Flux("SERCA", v_cyt_to_A_sr=307., density=75., scale=1.0)

## Temperature
T = 298.

## Calculate temperature dependent part
c0_T = -36./(87680.*exp(0.12*(-134.*T + 29600)/T) + 0.00133);

## Assign local variable from caller
Ca_i_2 = u0
Ca_SR_2  = u1

## Raise to the power of 2
Ca_SR_2 *= Ca_SR_2
Ca_i_2 *= Ca_i_2
  
## Calculate scaling factors
serca_scale = 1e-3           # from s to ms
serca_scale *= 2             # from cycles to Ca
serca_scale *= density       # From # to umole/(cytosole l)
serca_scale *= v_cyt_to_A_sr # From density to area distribution
serca_scale *= scale         # Arbitrary scale factor
serca_scale /= h             # Making the area integration dA/dV


## Applying CaSR and CaCyt dependencies
d_SERCA = serca_scale*(571000000.*Ca_i_2 + c0_T*Ca_SR_2)/ \
          (106700000.*Ca_i_2 + 182.5*Ca_SR_2 + 5.35)*dt
          
## Test expression. Should warning be raised.
# test_expr = d_SERCA/2.3 + sin(2.3)

# Here starts another flux definition. No need at this time
#
# ArgScalarParam is used to distinguish between parameters that values
# are read from file or set for command line arguments
Flux("RyR", "RyR.mm", Kd_open=127.92, k_min_open=1e-4, k_max_open=0.7, n_open=2.8, 
     Kd_close=62.5, k_min_close=0.9, k_max_close=10.0, n_close=-0.5,
     i_unitary=0.5, t_close_ryrs=ArgCloseParam())

# Flux is described as:
#
#   J_ryr = g_c/h^3(u0-u1)
#
# where:
#
#   g_c = K*i_c/(U0-U1)
#
# Here i_c is the unitary current through an open channel in pA, when it opens,
# U1 and U0 is the initial concentration when this current is measured, and K 
# a constant:
#
#   K = 10^15/(z*F)
#
# Here z is the valence of the current (2) and F Faraday's constant. 
#
# So with explicit Euler we have:
#
#   J_ryr = g_c/h^3*(u0-u1)
#   u0 -= dt*J_RyR;
#   u1 += dt*J_RyR
#
# With K so large this scheme will not be stable so we need to solve it 
# analytically with the following scheme:
#
#   c0 = (u1 + u0)/2
#   c1 = (u1 - u0)/2
#   u1 = c0 + c1*K_0 
#   u0 = c0 - c1*K_0
#
# where K_0 is given by:
#
#   K_0 = exp(-2*dt*g_c/h^3)

i_c = i_unitary

# g_c = 10^15/(z*F)*i_c/(u1-u0)
# K_0 = exp(-2*dt*g_c/h**3)
  
K_0 = exp(-2*dt*1e15/(2*96485.)*i_c/(1300.-0.1)/(h*h*h));
  
# The resulting analytic flux
d_RyR = (u0-u1)*(1-K_0)/2;

Flux("LCC", one=1.0)

d_LCC = one*0.5
