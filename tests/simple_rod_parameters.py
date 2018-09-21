from generate_parameters import *

float_type = np.float64
domains = ["nsr", "jsr", "cleft", "cyt"]
double = "" if float_type == np.float32 else "_double"

domain_species = [DomainSpecies("nsr", "Ca", 1300., 60.e3, fixed=True),
                  DomainSpecies("jsr", "Ca", 1300., 60.e3),
                  DomainSpecies("cleft", "Ca", 0.14, 350.e3),
                  DomainSpecies("cyt", "Ca", 0.14, 350.e3, fixed=True),
                  ]
buffers = []
ryr = RyR()
write_species_params("simple_rod_parameters{}.h5".format(double),
                     domains, domain_species, buffers, [ryr])
