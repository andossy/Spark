################################################################################
# RyR.sm file - part of serca_ryr.flux file
#
# In this scope one can use exactly the same variables like in the retaled
# *Flux* object. Use also emporary variables if necessary to desribes rate
# functions.

# *species_at_boudary* is a local variable in init and eval functions
# from BoundaryFluxes.h. The name is parametrized. The index *i* states that
# the same calculations should be done for all states. Meaning: generate
# a loop.
################################################################################


# Here we used variable *i* which the code generator translates into a for-loop
# for i=0, ..., ryr->states.N
c0 = species_at_boundary[i*2+1]

# Compute intermediates for the stochastic evaluation
ko = Max(Min((c0/Kd_open)**n_open, k_max_open), k_min_open)
kc = Max(Min((c0/Kd_close)**n_close, k_max_close), k_min_close)

# Rate functions. rate[A,B] = p reads as change Markov model from state B into
# into state A with the rate p.
rates[0,1] = kc
rates[1,0] = ko

# States values. states_values[A] = 1 reads as: state A is supposed to be 
# considered as open. 
states_values[1] = 1

