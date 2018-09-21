import sys
import shutil
import os
import h5py
import numpy as np
from collections import OrderedDict

# Typde defs
domain_id = np.uint8
float_type = np.float32
float_type = np.float64

class parameter_file:
    def __init__(self, filename):
        self.old_file = None
        self.filename = filename

    def __enter__(self):

        # Open a h5 file
        if os.path.isfile(self.filename):
            self.old_file = "_old_"+self.filename
            shutil.move(self.filename, self.old_file)

        self.f = h5py.File(self.filename)
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()
        if type is None:
            if self.old_file:
                os.unlink(self.old_file)
        else:
            shutil.move(self.old_file, self.filename)

class DomainSpecies:
    def __init__(self, domain, species, init=1.0, sigma=0., fixed=False):

        assert "-" not in species
        assert isinstance(domain, str)
        assert isinstance(species, str)
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
        return self.species+self.domain

class BufferReaction:
    def __init__(self, domain, species, bound_species, tot, k_on, k_off):
        assert isinstance(domain, str)
        assert isinstance(species, str)
        assert isinstance(bound_species, str)
        assert all(isinstance(value, float) and value > 0 for value in [tot, k_on, k_off])

        self.domain = domain
        self.species = species
        self.bound_species = bound_species
        self.tot = tot
        self.k_on = k_on
        self.k_off = k_off

    @property
    def kd(self):
        return self.k_off/self.k_on

class RyR:
    def __init__(self, Kd_open=127.92, k_min_open=1.e-4, k_max_open=0.7, n_open=2.8,
                 Kd_close=62.5, k_min_close=0.9, k_max_close=10., n_close=-0.5,
                 i_unitary=0.5):

        self.Kd_open = Kd_open
        self.k_min_open = k_min_open
        self.k_max_open = k_max_open
        self.n_open = n_open

        self.Kd_close = Kd_close
        self.k_min_close = k_min_close
        self.k_max_close = k_max_close
        self.n_close = n_close

        self.i_unitary = i_unitary
        self.boundary = "ryr"
        self.species = "Ca"

class SERCA:
    def __init__(self, v_cyt_to_A_sr=307., density=75., scale=1.0):
        self.v_cyt_to_A_sr = v_cyt_to_A_sr
        self.density = density
        self.scale = scale
        self.boundary = "serca"
        self.species = "Ca"

def write_species_params(filename, domains,
                         domain_species, buffer_reactions=None,
                         boundary_fluxes=None):

    boundary_fluxes = boundary_fluxes or []
    buffer_reactions = buffer_reactions or []
    assert isinstance(filename, str)

    # domain_species is a list with DomainSpecies
    assert isinstance(domain_species, list)
    assert all(isinstance(ds, DomainSpecies) for ds in domain_species)

    # Collect all distinct species and domains
    all_domains = domains
    species, domains = OrderedDict(), OrderedDict()
    diffusive_species = []
    for ds in domain_species:
        if ds.species not in species:
            species[ds.species] = []

        species[ds.species].append(ds.domain)
        assert ds.domain in all_domains, "expected {} to be in one of the "\
               "original domains".format(ds.species)

        if ds.domain not in domains:
            domains[ds.domain] = dict(species=OrderedDict(),
                                      buffers=OrderedDict(),
                                      diffusive=OrderedDict())
        domains[ds.domain]["species"][ds.species] = ds

        # Store diffusive spieces
        if ds.sigma>0:
            domains[ds.domain]["diffusive"][ds.species] = ds
            diffusive_species.append(ds.species)

    # buffer_reactions is a list with BufferReactions
    assert isinstance(buffer_reactions, list)
    assert all(isinstance(br, BufferReaction) for br in buffer_reactions)
    for br in buffer_reactions:
        assert br.domain in domains, "{}!={}".format(br.domain, domains)
        assert br.species in species, "{}!={}".format(br.species, species)
        assert br.bound_species in species, "{}!={}".format(br.bound_species, species)
        assert br.species in domains[br.domain]["species"], "{}!={}".format(\
            br.bound_species, omains[br.domain]["species"].keys())
        assert br.bound_species in domains[br.domain]["species"], "{}!={}".format(\
            br.bound_species, domains[br.domain]["species"].keys())
        domains[br.domain]["buffers"][br.species] = br

        # Update initial values of buffers
        b = domains[br.domain]["species"][br.species]
        s = domains[br.domain]["species"][br.bound_species]
        b.init = br.tot*s.init/(s.init+br.kd)

    # Lump species name in different domains
    max_num_species = max(len(domains[domain]["species"]) for domain in domains)
    min_num_species = min(len(domains[domain]["species"]) for domain in domains)

    # FIXME: Improve deduction of distinct species.
    # 1) Start with the domain with most species.
    # 2) Find all diffusive species. Iterate over the other domains and try
    #    bining species together.
    # 3) Need to do 2) in 2 sweeps.
    #    a) Identify what species in the domain have the same(ish) name and bin
    #       these together
    #    b) Bin the other species together. If there are none diffusive species. Move
    #       them to the end
    #print "max_num_species", max_num_species
    #print "min_num_species", min_num_species

    distinct_species = [set() for i in range(min_num_species)]
    for domain in domains:
        sps = domains[domain]["species"].keys()
        for i in range(min_num_species):
            distinct_species[i].add(sps[i])

    #print "distinct_species", distinct_species
    species_map = OrderedDict()
    for old_species, new_species in zip(distinct_species, \
                                        ["-".join(sps) for sps in distinct_species]):
        for osp in old_species:
            species_map[osp] = new_species

    # Update all species names
    for domain in domains:
        for what in ["species", "buffers", "diffusive"]:
            new_species = OrderedDict()
            for os, ds in domains[domain][what].items():
                if os in species_map:
                    new_species[species_map[os]] = ds
                    ds.species = species_map[os]

                else:
                    new_species[ds.species] = ds

                if isinstance(ds, BufferReaction) and ds.bound_species in species_map:
                    ds.bound_species = species_map[ds.bound_species]

            domains[domain][what] = new_species

    #for dom, domsps in domains.items():
    #    print
    #    print dom, "species"
    #    for sp, domsp in domsps["species"].items():
    #        print dom, sp, domsp
    #
    #    print
    #    print dom, "diffusive"
    #    for sp, domsp in domsps["diffusive"].items():
    #        print dom, sp, domsp
    #
    #    print
    #    print dom, "buffers"
    #    for sp, domsp in domsps["buffers"].items():
    #        print dom, sp, domsp

    #print species_map
    #print species_list

    updated_species = OrderedDict()
    for sp, sp_domains in species.items():
        if sp in species_map:
            new_sp = species_map[sp]
            if new_sp in updated_species:
                for domain in sp_domains:
                    if domain not in updated_species[new_sp]:
                        updated_species[new_sp].append(domain)
            else:
                updated_species[new_sp] = species[sp][:]
            if sp in diffusive_species:
                diffusive_species.remove(sp)
                diffusive_species.append(new_sp)
        else:
            updated_species[sp] = sp_domains[:]

    # FIXME: Not safe sorting as it does not take a look into all domains...
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

    species = updated_species
    species_list = sorted(species.keys(), cmp=species_cmp)
    domain_list = sorted(all_domains, cmp=domain_cmp)

    print "New species:", species_list
    print "Check that these make sense as we have a faulty selection algorithm."

    # Open parameter h5 file
    with parameter_file(filename) as f:

        # Extract Boundary fluxes
        use_ryr = False
        use_serca = False
        for flux in boundary_fluxes:
            if isinstance(flux, RyR):
                ryr = flux
                use_ryr = True
            elif isinstance(flux, SERCA):
                serca = flux
                use_serca = True

        f.attrs.create("use_ryr", use_ryr, dtype=domain_id)
        f.attrs.create("use_serca", use_serca, dtype=domain_id)

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

        # Add domains
        f.attrs.create("num_domains", len(domain_list), dtype=domain_id) # cyt, jsr, tt
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

                if dom in domains and sp in domains[dom]["species"]:
                    sigma = domains[dom]["species"][sp].sigma
                    init = domains[dom]["species"][sp].init
                    if domains[dom]["species"][sp].fixed:
                        fixed.extend([indd, inds])
                else:
                    sigma = 0.0
                    init = 0.0

                inits.append(init)
                if sigma > 0:
                    num_diffusive += 1
                    diffusive.append(inds)
                    sigmas.append(sigma)

            if dom in domains:
                num_buff = len(domains[dom]["buffers"])
                for indb, buff in enumerate(sorted(domains[dom]["buffers"].values(),
                                                   cmp=lambda a,b:cmp(
                                                       species_list.index(a.species),
                                                       species_list.index(b.species)))):
                    bg = g.create_group("buffer_{}".format(indb))
                    buff_sp = [species_list.index(buff.species), \
                               species_list.index(buff.bound_species)]
                    bg.attrs.create("species", np.array(buff_sp, dtype=domain_id))
                    bg.attrs.create("k_off", buff.k_off, dtype=float_type)
                    bg.attrs.create("k_on", buff.k_on, dtype=float_type)
                    bg.attrs.create("tot", buff.tot, dtype=float_type)
            else:
                num_buff = 0

            g.attrs.create("num_buffers", num_buff, dtype=domain_id)
            g.attrs.create("num_diffusive", num_diffusive, dtype=domain_id)
            g.attrs.create("diffusive", np.array(diffusive, dtype=domain_id))
            g.attrs.create("sigma", np.array(sigmas, dtype=float_type))
            g.attrs.create("init", np.array(inits, dtype=float_type))

        f.attrs.create("num_fixed_domain_species", len(fixed)/2, dtype=domain_id)
        f.attrs.create("fixed_domain_species", np.array(fixed, dtype=domain_id))

        # Save convolution constants in nm
        s_x = 20000/np.log(2); s_y = s_x; s_z = 80000/np.log(2)
        f.attrs.create("convolution_constants", np.array([s_x, s_y, s_z], dtype=float_type))

if __name__ == "__main__":

    single_species = 0
    two_species = 0
    domain_species = [DomainSpecies("cyt", "Ca", 0.14, sigma=350.e3),
                      DomainSpecies("cyt", "Fluo3", sigma=42.e3),
                      DomainSpecies("cyt", "CMDN", sigma=22.e3),
                      DomainSpecies("cyt", "ATP", sigma=140.e3),
                      DomainSpecies("cyt", "TRPN"),

                      DomainSpecies("cleft", "Ca", 0.14, sigma=350.e3*0.4),
                      DomainSpecies("cleft", "Fluo3", sigma=42.e3*0.4),
                      DomainSpecies("cleft", "CMDN", sigma=22.e3*0.4),
                      DomainSpecies("cleft", "ATP", sigma=140.e3*0.4),
                      DomainSpecies("cleft", "TRPN"),

                      DomainSpecies("jsr", "Ca", 1300., 60.e3),
                      DomainSpecies("jsr", "Fluo5", sigma=10.e3),
                      DomainSpecies("jsr", "CSQN"),

                      DomainSpecies("nsr", "Ca", 1300., 60.e3),
                      DomainSpecies("nsr", "Fluo5", sigma=10.e3),
                      DomainSpecies("nsr", "CSQN"),
                      ]

    buffers = [BufferReaction("cyt", "CMDN", "Ca", 24., 34e-3, 238.e-3),
               BufferReaction("cyt", "ATP", "Ca", 455., 255e-3, 45.),
               BufferReaction("cyt", "Fluo3", "Ca", 25., 255e-3, 45.),
               BufferReaction("cyt", "TRPN", "Ca", 70., 32.7e-3, 19.6e-3),

               BufferReaction("cleft", "CMDN", "Ca", 24., 34e-3, 238.e-3),
               BufferReaction("cleft", "ATP", "Ca", 455., 255e-3, 45.),
               BufferReaction("cleft", "Fluo3", "Ca", 25., 255e-3, 45.),
               BufferReaction("cleft", "TRPN", "Ca", 47., 32.7e-3, 19.6e-3),

               BufferReaction("jsr", "Fluo5", "Ca", 25., 255e-3, 45.),
               BufferReaction("jsr", "CSQN", "Ca", 30.e3, 102e-3, 65.),

               BufferReaction("nsr", "Fluo5", "Ca", 25., 255e-3, 45.),
               BufferReaction("nsr", "CSQN", "Ca", 6.e3, 102e-3, 65.),
               ]

    domains = ["cyt", "cleft", "jsr", "nsr", "tt"]
    #domains = ["cyt", "cleft", "jsr", "tt"]
    double = "" if float_type == np.float32 else "_double"
    suffix = ""

    if single_species:
        domain_species = [DomainSpecies("cyt", "Ca", 0.14, 350.e3),
                          DomainSpecies("cleft", "Ca", 0.14, 350.e3),
                          DomainSpecies("jsr", "Ca", 1300., 60.e3),
                          ]
        buffers = []
        suffix = "_Ca"
    elif two_species:
        domain_species = [DomainSpecies("cyt", "Ca", 0.14, 350.e3),
                          DomainSpecies("cyt", "Fluo3", sigma=42.e3),

                          DomainSpecies("cleft", "Ca", 0.14, 350.e3),
                          DomainSpecies("cleft", "Fluo3", sigma=42.e3),

                          DomainSpecies("jsr", "Ca", 1300., 60.e3),
                          DomainSpecies("jsr", "Fluo5", sigma=10.e3),
                          ]

        buffers = [BufferReaction("cyt", "Fluo3", "Ca", 25., 255e-3, 45.),
                   BufferReaction("cleft", "Fluo3", "Ca", 25., 255e-3, 45.),
                   BufferReaction("jsr", "Fluo5", "Ca", 25., 255e-3, 45.),
                   ]
        suffix = "_Ca_Fluo"

    #ryr = RyR(Kd_open=30., Kd_close=20., i_unitary=0.125)
    #ryr = RyR(Kd_open=50., Kd_close=20., i_unitary=0.125)
    #ryr = RyR(Kd_open=80., Kd_close=50.)
    ryr = RyR(Kd_open=105., Kd_close=62.5, k_min_open=1.e-4*0.05)
    #ryr = RyR(Kd_open=90., Kd_close=62.5)
    #ryr = RyR(Kd_open=80., Kd_close=62.5)
    #ryr = RyR(Kd_open=60., Kd_close=30.)
    #ryr = RyR(Kd_open=50., Kd_close=20.)
    #ryr = RyR(Kd_open=40., Kd_close=20.)
    #ryr = RyR(Kd_open=30., Kd_close=20.)
    #ryr = RyR(k_min_open=.7, k_max_open=.7, k_min_close=0.9, k_max_close=0.9)

    serca = SERCA()
    #write_species_params("parameters_flat_1{}{}.h5".format(double, suffix),
    #                     domains, domain_species, buffers, [ryr, serca])
    write_species_params("parameters_cleft_kdo_{}_kdc_{}_i_{}{}{}.h5".format(
        int(ryr.Kd_open), int(ryr.Kd_close), float(ryr.i_unitary), double, suffix),
                         domains, domain_species, buffers, [ryr, serca])

