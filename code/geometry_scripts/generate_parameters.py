import sys
import numpy as np
from collections import OrderedDict
from h5files import ParameterFile
import logging

# Global parameters and type defs
DOMAINS = ["cyt", "cleft", "jsr", "nsr", "tt"]
domain_id = np.uint8
float_type = np.float32
float_type = np.float64

class UndefinedSpeciesError(Exception):
    pass

class UndefinedDomainError(Exception):
    pass

class SpeciesDomainMismatchError(Exception):
    pass

class DomainSpecies(object):
    """Container for a specific species in a given domain.

    Parameters
    ----------
    domain : str
        Name of the domain the species resides in
    species : str
        Name of the species
    init : float, optional
        Initial concentration of the species, must be positive
    sigma : float, optional
        Diffusivity. Defaults to 0, meaning the species is non-diffusive
    fixed : bool, optional
        ???
    """
    def __init__(self, domain, species, init=1.0, sigma=0., fixed=False):
        # Sanity checks on input
        assert isinstance(domain, str), "domain must be given as str"
        assert domain in DOMAINS, "domain not recognized"
        assert isinstance(species, str), "species must be given as str"
        assert "-" not in species, "- is not allowed in species name"
        assert isinstance(init, float)
        assert isinstance(sigma, float)
        assert init > 0.
        assert sigma >= 0.0

        self.species = species
        self.domain = domain
        self.sigma = sigma
        self.init = init
        self.fixed = fixed

    @property
    def domain_name(self):
        return self.species + self.domain

    @property
    def diffusive(self):
        """True if the species is diffusive in its domain."""
        return self.sigma > 0.

    def __str__(self):
        return "DomainSpecies({}, {}, {})".format(self.domain, 
                                                  self.species,
                                                  "diffusive" if self.diffusive
                                                  else "non-diffusive")

class BufferReaction(object):
    """Container for a buffer reaction in a given domain.

    Parameters
    ----------
    domain : str
        Name of domain where buffer reaction is found
    species : str
        Name of buffer species
    bound_species : str
        Name of species being bound by buffer
    tot : float
        Total concentration of buffer
    k_on : float
        Association rate coefficient
    k_off : float
        Dissociation rate coefficient
    """
    def __init__(self, domain, species, bound_species, tot, k_on, k_off):
        # Sanity checks on input
        assert isinstance(domain, str), "domain must be given as str"
        assert domain in DOMAINS, "domain not recognized"
        assert isinstance(species, str)
        assert isinstance(bound_species, str)
        assert all(isinstance(v, float) and v > 0 for v in [tot, k_on, k_off])
        
        self.domain = domain
        self.species = species
        self.bound_species = bound_species
        self.tot = tot
        self.k_on = k_on
        self.k_off = k_off

    @property
    def kd(self):
        """Equilibrium binding concstant."""
        return self.k_off/self.k_on

    def __str__(self):
        return "BufferReaction({}, {} <-> {})".format(self.domain, 
                                                   self.species,
                                                   self.bound_species)

class BoundaryFlux(object):
    """Container for fluxes between domains."""
    pass

class RyR(BoundaryFlux):
    """Container class for a ryanodine receptor.

    Notes
    -----
    ??? 

    Parameters
    ----------
    Kd_open : float, optional
        Equilibrium binding constant in open state
    Kd_close : float, optional
        Equilibrium binding constant in closed state
    k_min_open : float, optional
        ???
    k_max_open : float, optional
        ???
    k_min_close : float, optional
        ???
    k_max_close : float, optional
        ???
    n_open : float, optional
        ???
    n_close : float, optional
        ???
    i_unitary : float, optional
        ???
    """
    def __init__(self, Kd_open=127.92, 
                       k_min_open=1.e-4, 
                       k_max_open=0.7, 
                       n_open=2.8, 
                       Kd_close=62.5, 
                       k_min_close=0.9, 
                       k_max_close=10., 
                       n_close=-0.5,
                       i_unitary=0.5):
        self.boundary = "ryr"
        self.species = "Ca"
        self.i_unitary = i_unitary
        
        self.Kd_open = Kd_open
        self.k_min_open = k_min_open
        self.k_max_open = k_max_open
        self.n_open = n_open
        
        self.Kd_close = Kd_close
        self.k_min_close = k_min_close
        self.k_max_close = k_max_close
        self.n_close = n_close


class SERCA(BoundaryFlux):
    """Container for SERCA channels.

    Parameters
    ----------
    v_cyt_to_A_sr : float, optional
        ???
    density : float, optional
        ???
    scale : float, optional
        ???
    """
    def __init__(self, v_cyt_to_A_sr=307., density=75., scale=1.0):
        assert all(isinstance(v, float) for v in [v_cyt_to_A_sr, density, scale])
        self.boundary = "serca"
        self.species = "Ca"
        self.v_cyt_to_A_sr = v_cyt_to_A_sr
        self.density = density
        self.scale = scale


def write_species_params(filename, domains, domain_species, 
                         buffer_reactions=None, boundary_fluxes=None):
    """

    Parameters
    ----------
    filename : str
        File to write the parameter data to.
    domains : list of str
    domain_species : list of DomainSpecies objects
    buffer_reactions : list of BufferReaction objects
    boundary_fluxes : list of Flux objects
    """
    boundary_fluxes if boundary_fluxes is not None else []
    buffer_reactions = buffer_reactions if buffer_reactions is not None else []

    # Type checks
    assert isinstance(filename, str)
    assert isinstance(domains, list)
    assert isinstance(domain_species, list)
    assert isinstance(buffer_reactions, list)
    assert isinstance(boundary_fluxes, list)
    assert all(isinstance(d, str) for d in domains)
    assert all(isinstance(ds, DomainSpecies) for ds in domain_species)
    assert all(isinstance(br, BufferReaction) for br in buffer_reactions)
    assert all(isinstance(bf, BoundaryFlux) for bf in boundary_fluxes)
    
    all_domains = domains #???
    species = OrderedDict()
    domains = OrderedDict()
    diffusive_species = []

    for ds in domain_species:
        assert ds.domain in all_domains, "domain {} is not among the "\
               "original domains".format(ds.species)
        
        # Add to species dict
        if ds.species not in species:
            species[ds.species] = []
        species[ds.species].append(ds.domain)

        # Add to domain dict
        if ds.domain not in domains:
            domains[ds.domain] = dict(species=OrderedDict(),
                                       buffers=OrderedDict(),
                                       diffusive=OrderedDict())
        domains[ds.domain]['species'][ds.species] = ds

        if ds.diffusive:
            domains[ds.domain]['diffusive'][ds.species] = ds
            if ds.species not in diffusive_species:
                diffusive_species.append(ds.species)

    # Checking that BufferReaction and DomainSpecies objects are consistent
    for br in buffer_reactions:
        if br.domain not in domains:
            e = "Buffer domain '{}' not recognized".format(br.domain)
            raise UndefinedDomainError(e)
        if br.species not in species:
            e = "Buffer species '{}' not recognized".format(br.species)
            raise UndefinedSpeciesError(e)
        if br.bound_species not in species:
            e = "Bound species '{}' not recognized".format(br.species)
            raise UndefinedSpecieserror(e)
        if br.species not in domains[br.domain]['species']:
            print domains[br.domain]['species']
            e = "'{}' not found in '{}'".format(br.species, br.domain)
            raise SpeciesDomainMismatchError(e)
        if br.bound_species not in domains[br.domain]['species']:
            e = "'{}' not found in '{}'".format(br.bound_species, br.domain)
            raise SpeciesDomainMismatchError(e)

        domains[br.domain]['buffers'][br.species] = br

        # Update initial value of buffers
        b = domains[br.domain]['species'][br.species]
        s = domains[br.domain]['species'][br.bound_species]
        b.init = br.tot*s.init/(s.init + br.kd)

    # Lump species name in different domains
    max_num_species = max(len(domains[domain]['species']) for domain in domains)
    min_num_species = min(len(domains[domain]['species']) for domain in domains)
    distinct_species = [set() for i in range(min_num_species)]
    for domain in domains:
        sps = domains[domain]['species'].keys()
        for i in range(min_num_species):
            distinct_species[i].add(sps[i])
    
    # Make a map linking old names to new names
    species_map = OrderedDict()
    for old_species, new_species in zip(distinct_species, \
                                       ["-".join(s) for s in distinct_species]):
        for osp in old_species:
            species_map[osp] = new_species

    # Log info about species lumping
    logging.debug("The following species are lumped")
    for os, ns in species_map.items():
        if os != ns:
            logging.debug("{} -> {}".format(os, ns))

    # Update all species names from the map
    for domain in domains:
        for stype in ['species', 'buffers', 'diffusive']:
            new_species = OrderedDict()
            for sname, sobject in domains[domain][stype].items():
                if sname in species_map:
                    new_species[species_map[sname]] = sobject
                    sobject.species = species_map[sname]
                else:
                    new_species[sname] = sobject
                if isinstance(sobject, BufferReaction):
                    if sobject.bound_species in species_map:
                      sobject.bound_species = species_map[sobject.bound_species]
            domains[domain][stype] = new_species
    
    # Log info for debugging
    species_info = 'Domain Species:\n\t\t  {}'.format("\n\t\t  ".join(
                    (str(ds) for domain in domains 
                             for ds in domains[domain]['species'].values())))
    diffusion_info = 'Domain Species:\n\t\t  {}'.format("\n\t\t  ".join(
                    (str(ds) for domain in domains 
                             for ds in domains[domain]['diffusive'].values())))
    buffer_info = 'Buffer Reactions:\n\t\t  {}'.format("\n\t\t  ".join(
                    (str(ds) for domain in domains 
                             for ds in domains[domain]['buffers'].values())))
    logging.debug(species_info)
    logging.debug(buffer_info)

    # Change species dict to reflect lumping
    updated_species = OrderedDict()
    for os, os_domains in species.items():
        if os not in species_map:
            updated_species[os] = os_domains[:]
        else:
            ns = species_map[os]
            if ns not in updated_species:
                updated_species[ns] = os_domains[:]
            else:
                for domain in os_domains:
                    if domain not in updated_species[ns]:
                        updated_species[ns].append(domain)

            if os in diffusive_species:
                index = diffusive_species.index(os)
                diffusive_species.pop(index)
                diffusive_species.insert(index, ns)
    species = updated_species

    def species_cmp(a, b):
        if a not in diffusive_species and b not in diffusive_species:
            return 0
        elif a in diffusive_species and b in diffusive_species:
            return 0
        elif a in diffusive_species:
            return -1
        else:
            return 1

    def domain_cmp(a, b):
        if a not in domains and b not in domains:
            return 0
        elif a in domains and b in domains:
            return cmp(domains.keys().index(a), domains.keys().index(b))
        elif a in domains:
            return -1
        else:
            return 1

    species_list = sorted(species.keys(), cmp=species_cmp)
    domain_list = sorted(all_domains, cmp=domain_cmp)

    logging.warning("New species: {}".format(species_list))
    logging.warning("Check that these make sense as we "
                    "have a faulty selection algorithm.")
    
    # Open parameter h5 file and write data
    with ParameterFile(filename) as f:
        # Extract boundary fluxes
        use_ryr = False
        use_serca = False
        for flux in boundary_fluxes:
            if isinstance(flux, RyR):
                ryr = flux
                use_ryr = True
            elif isinstance(flux, SERCA):
                serca = flux
                use_serca = True

        f.attrs.create("use_ryr", use_ryr, dtype=np.uint8)
        f.attrs.create("use_serca", use_serca, dtype=np.uint8)

        if use_ryr:
            g_ryr = f.create_group("ryr")
            g_ryr.attrs.create("Kd_open", ryr.Kd_open, dtype=float_type)
            g_ryr.attrs.create("k_min_open", ryr.k_min_open, dtype=float_type)
            g_ryr.attrs.create("k_max_open", ryr.k_max_open, dtype=float_type)
            g_ryr.attrs.create("n_open", ryr.n_open, dtype=float_type)

            g_ryr.attrs.create("Kd_close", ryr.Kd_close, dtype=float_type)
            g_ryr.attrs.create("k_min_close", ryr.k_min_close, dtype=float_type)
            g_ryr.attrs.create("k_max_close", ryr.k_max_close, dtype=float_type)
            g_ryr.attrs.create("n_close", ryr.n_close, dtype=float_type)

            g_ryr.attrs.create("i_unitary", ryr.i_unitary, dtype=float_type)
            g_ryr.attrs.create("boundary", ryr.boundary)
            g_ryr.attrs.create("species", ryr.species)

        if use_serca:
            g_serca = f.create_group("serca")
            g_serca.attrs.create("v_cyt_to_A_sr", serca.v_cyt_to_A_sr,
                                 dtype=float_type)
            g_serca.attrs.create("density", serca.density, dtype=float_type)
            g_serca.attrs.create("scale", serca.scale, dtype=float_type)
            g_serca.attrs.create("v_cyt_to_A_sr", serca.v_cyt_to_A_sr, dtype=float_type)
            g_serca.attrs.create("boundary", serca.boundary)
            g_serca.attrs.create('species', serca.species)

        # Add domains
        f.attrs.create("num_domains", len(domain_list), dtype=domain_id)
        for num, dn in enumerate(domain_list):
            f.attrs.create("domain_name_{}".format(num), dn)

        # Add species
        f.attrs.create("num_species", len(species_list), dtype=domain_id)
        for inds, sp in enumerate(species_list):
            f.attrs.create("species_name_{}".format(inds), sp)

        # Iterate over the domains and add species and buffer information
        fixed = []
        for indd, dom in enumerate(domain_list):
            g = f.create_group(dom)
            num_diffusive = 0
            diffusive = []
            sigmas = []
            inits = []
            for inds, sp in enumerate(species_list):
                if dom in domains and sp in domains[dom]['species']:
                    sigma = domains[dom]['species'][sp].sigma
                    init = domains[dom]['species'][sp].init
                    if domains[dom]['species'][sp].fixed:
                        fixed.extend([indd,inds])
                else:
                    sigma = 0.0
                    init = 0.0

                inits.append(init)
                if sigma > 0:
                    num_diffusive += 1
                    diffusive.append(inds)
                    sigmas.append(sigma)
                
            if dom in domains:
                num_buff = len(domains[dom]['buffers'])
                for indb, buff in enumerate(sorted(domains[dom]['buffers'].values(),
                                                   cmp=lambda a,b:cmp(
                                                       species_list.index(a.species),
                                                       species_list.index(b.species)))):
                    bg = g.create_group("buffer_{}".format(indb))
                    buff_sp = [species_list.index(buff.species), \
                               species_list.index(buff.bound_species)]
                    bg.attrs.create('species', np.array(buff_sp, dtype=domain_id))
                    bg.attrs.create("k_off", buff.k_off, dtype=float_type)
                    bg.attrs.create("k_on", buff.k_on, dtype=float_type)
                    bg.attrs.create("tot", buff.tot, dtype=float_type)
            else:
                num_buff = 0
            
            g.attrs.create("num_buffers", num_buff, dtype=domain_id)
            g.attrs.create("num_diffusive", num_diffusive, dtype=domain_id)
            g.attrs.create('diffusive', np.array(diffusive, dtype=domain_id))
            g.attrs.create("sigma", np.array(sigmas, dtype=float_type))
            g.attrs.create("init", np.array(inits, dtype=float_type))

        f.attrs.create("num_fixed_domain_species", len(fixed)/2, dtype=domain_id)
        f.attrs.create("fixed_domain_species", np.array(fixed, dtype=domain_id))

        # Save convolution constants in nm
        s_x = 20000/np.log(2); s_y = s_x; s_z = 80000/np.log(2)
        #s_x = 5000/np.log(2); s_y = s_x; s_z = 5000/np.log(2)
        #s_x = 12/np.log(2); s_y = s_x; s_z = 12/np.log(2)
        f.attrs.create("convolution_constants", np.array([s_x, s_y, s_z], dtype=float_type))

if __name__ == "__main__":
    logging.getLogger().setLevel(20)

    f = 1.0

    domain_species = [DomainSpecies("cyt", "Ca", 0.14, sigma=220.e3),
                      DomainSpecies("cyt", "Fluo3", sigma=42.e3),
                      DomainSpecies("cyt", "CMDN", sigma=22.e3),
                      DomainSpecies("cyt", "ATP", sigma=140.e3),
                      DomainSpecies("cyt", "TRPN"),
                      DomainSpecies("cleft", "Ca", 0.14, sigma=220.e3*f),
                      DomainSpecies("cleft", "Fluo3", sigma=42.e3*f),
                      DomainSpecies("cleft", "CMDN", sigma=22.e3*f),
                      DomainSpecies("cleft", "ATP", sigma=140.e3*f),
                      DomainSpecies("cleft", "TRPN"),

                      DomainSpecies("jsr", "Ca", 10.0, 73.3e3),
                      DomainSpecies("jsr", "Fluo5", sigma=8.e3),
                      DomainSpecies("jsr", "CSQN"),

                      DomainSpecies("nsr", "Ca", 1300., 0.01*73.3e3),
                      DomainSpecies("nsr", "Fluo5", sigma=8.e3),
                      DomainSpecies("nsr", "CSQN")]

    buffers = [BufferReaction("cyt", "CMDN", "Ca", 24., 34e-3, 238.e-3),
               BufferReaction("cyt", "ATP", "Ca", 455., 255e-3, 45.),
               BufferReaction("cyt", "Fluo3", "Ca", 25., 10*110e-3, 10*110e-3),
               BufferReaction("cyt", "TRPN", "Ca", 70., 32.7e-3, 19.6e-3),
               
               BufferReaction("cleft", "CMDN", "Ca", 24., 34e-3, 238.e-3),
               BufferReaction("cleft", "ATP", "Ca", 455., 255e-3, 45.),
               BufferReaction("cleft", "Fluo3", "Ca", 25.,10*110e-3, 10*110e-3),
               BufferReaction("cleft", "TRPN", "Ca", 47., 32.7e-3, 19.6e-3),

               #BufferReaction("jsr", "Fluo5", "Ca", 25., 250e-6, 100e-3),
               BufferReaction("jsr", "Fluo5", "Ca", 25.e-3, 110e-3, 110e-3),
               BufferReaction("jsr", "CSQN", "Ca", 30.e3, 102e-3, 65.),
               
               #BufferReaction("nsr", "Fluo5", "Ca",25., 250e-6, 100e-3),
               BufferReaction("nsr", "Fluo5", "Ca", 25.e-3, 110e-6, 110e-3),
               BufferReaction("nsr", "CSQN", "Ca", 6.e3, 102e-3, 65.)
               ]

    domains = ["cyt", "cleft", "jsr", "nsr", "tt"]

    double = "" if float_type == np.float32 else "_double"
    suffix = ""

    #ryr = RyR(Kd_open=80., Kd_close=50.)
    #ryr = RyR(Kd_open=105., Kd_close=62.5, k_min_open=1.e-4*0.05, i_unitary= 0.5)

    ryr = RyR(Kd_open=105., Kd_close=60, k_min_open=1.e-4*0.05, i_unitary= 0.5) # Kd_open = 60: shifted
    #ryr = RyR(Kd_open=105., Kd_close=30.0, k_min_open=1.e-4*0.05, i_unitary= 0.5)
    #ryr = RyR(Kd_open=90., Kd_close=62.5)
    
    serca = SERCA(density=1000.) #this pumps at 20uM/10ms, 700um/350ms, perhaps
    
    #write_species_params("parameters_flat_1{}{}.h5".format(double, suffix),
    #                     domains, domain_species, buffers, [ryr, serca])

    #write_species_params("parameters_kdo_{}_kdc_{}_i_{}{}{}.h5".format(        int(ryr.Kd_open), int(ryr.Kd_close), float(ryr.i_unitary), double, suffix),                         domains, domain_species, buffers, [ryr, serca])
    
    write_species_params("no_jsr_ca.h5", domains, domain_species, buffers, [ryr, serca])
    

