#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["parameters"]

from modelparameters.parameterdict import ParameterDict
from modelparameters.parameters import Param
#from string import lower

# Tempoorary parameter, it is defined here since the value is used twice in the
# below definition of parameters 

parameters = ParameterDict(

    # Output file name without suffixes
    output = Param("boundaryfluxes", description="Specified output file name"),
   
    # Code generation parameters
    code = ParameterDict(
        
        # Float precision
        float_precision = Param("REAL", description="Float precision in "\
                                "generated code"),
        
        # Typedef suffix                        
        typedef_suffix = Param("_t", description="Suffix that is added to "\
                               "structures names using typedef"),
        
        # String length variable
        string_length = Param("MAX_SPECIES_NAME", description="Varaible "\
                                  "name holding length of used strings"),

        # Parameters for generating structs representing fluxes
        flux = ParameterDict(
        
            # Flux struct name suffix
            suffix = Param("_model", description="Suffix that is to be added "\
                           "to the struct name"),
            
            # Flux typedef struct name suffix, will be set later
            typedef_suffix = Param("", description="Suffix that "\
                                   "is used with typedef"),
                                   
            # Parameter function that translates flux name defined in a .flux
            # file into variable name used in generated code.
            # This name should also be the same as the name appearing in the
            # parameters.h5 file
#            variable_function = Param(lower, description="A function handler "\
#                                     "that translates Flux name into Flux "\
#                                     "variable name"),
                                     
            # Name of parameters in a Flux structure
            params = Param("params", description="Name of temporary "\
                               "parameters from Flux structures"),
                               
            # Parameters for state variables
            states = Param("states", description="Name of ModelStates variable"),
                               
            # Name of species this flux is applied to
            species = Param("species_name", description="Name of species "\
                            "this flux is applied to"),
                            
            # Name of boundary this flux exis
            boundary = Param("boundary_name", description="Name of boundary "\
                             "this flux exists on")
        ), #end of flux
        
        # Parameters for Flux wrappers
        boundary_flux = ParameterDict(
        
            # Wrapper name
            var_name = Param("fluxes", description="Variable name of wrappers "\
                             "type"),
                             
            # Number of used fluxes
            num_of_fluxes = Param("num_of_used_fluxes", description="Name of variable "\
                                  "counting used fluxes"),
                                  
            # Map name
            map = Param("map", description="Name of map: boundary index -> "\
                        "flux info structure"),
                        
            # boundary map
            boundary_map = Param("boundary_map", description="Name of boundary "\
                                 "map")
        ) # end of boundary_flux
    ), # end of code
    
    # Symbol parameters, this part ispassed to flux object
    symbols = ParameterDict(
        
        states = ParameterDict(
        
            # States number variable
            length = Param("N", description="Variable denoting the number of "\
                           "states"),
                           
            variable = Param("s0", description="Variable name representing "\
                             "states")
        ), # end of states
        
        # Parameters for additional variables used in flux calculations
        variables = ParameterDict(
            
            # time-step
            dt = Param("dt", description="Name of time step argument"),
            
            # spatial step
            h = Param("h", description="Name of spatial step argument"),

            # local variables
            loc_one = Param("u0", description="First local variable in a "\
                            "flux function"),

            loc_two = Param("u1", description="Second local variable in a "\
                            "flux function"),
                            
            species = Param("species_at_boundary", description="Local variable "\
                            "in stochastic functions"),
                            
            iterate = Param("i", description="Local variable to iterate over states"),
            
            init = Param("number_init_"),
            open = Param("open_"),

            rand = Param("mt_ldrand()", description="C function used to generate "\
                            "random numbers")
        ) # end of variables
    ) # end of symbols
) # end of parameters


# Set up the left values
parameters.code.flux.typedef_suffix = parameters.code.flux.suffix + \
                                      parameters.code.typedef_suffix
