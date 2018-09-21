#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from collections import deque
from textwrap import dedent

from modelparameters.utils import check_arg, check_kwarg
from sympytools import ArgCloseParam

from printing import ccode
from base import *
from models import Flux
from options import parameters

class BaseCodeGenerator(object):
    """
    Base class for all code generators taken from modelparameters with several
    major changes applied.
    """

    # Class attributes
    language = "None"
    line_ending = ""
    closure_start = ""
    closure_end = ""
    line_cont = "\\"
    comment = "#"
    index = lambda x, i : "[{0}]".format(i)
    indent = 4
    indent_str = " "
    max_line_length = 79
    to_code = lambda self,b,c,d : None


    def __init__(self, params=None):
        super(BaseCodeGenerator, self).__init__()

        params = params or {}

        self.params = self.default_parameters()
        self.params.update(params)


    @staticmethod
    def default_parameters():
        return parameters.copy()


    @classmethod
    def indent_and_split_lines(cls, code_lines, indent=0, ret_lines=None, \
                               no_line_ending=False):
        """
        Combine a set of lines into a single string
        """

        _re_str = re.compile(".*\".*\".*")

        def _is_number(num_str):
            """
            A hack to check wether a str is a number
            """
            try:
                float(num_str)
                return True
            except:
                return False

        check_kwarg(indent, "indent", int, ge=0)
        ret_lines = ret_lines or []

        # Walk through the code_lines
        for line_ind, line in enumerate(code_lines):

            # If another closure is encountered
            if isinstance(line, list):

                # Add start closure sign if any
                if cls.closure_start and len(line) > 1 and code_lines[line_ind-1].find('case') == -1:
                    ret_lines.append(cls.indent*indent*cls.indent_str + \
                                     cls.closure_start)

                ret_lines = cls.indent_and_split_lines(\
                    line, indent+1, ret_lines)

                # Add closure if any
                if cls.closure_end and len(line) > 1 and code_lines[line_ind-1].find('case') == -1:
                    ret_lines.append(cls.indent*indent*cls.indent_str + \
                                     cls.closure_end)
                continue

            line_ending = "" if no_line_ending else cls.line_ending

            # Do not use line endings the line before and after a closure
            if line_ind + 1 < len(code_lines):
                if isinstance(code_lines[line_ind+1], list):
                    line_ending = ""

            # Check if we parse a comment line
            if len(line) > len(cls.comment) and cls.comment == \
               line[:len(cls.comment)]:
                is_comment = True
                line_ending = ""
            else:
                is_comment = False

            # Empty line
            if line == "":
                ret_lines.append(line)
                continue

            # Check for long lines
            if cls.indent*indent + len(line) + len(line_ending) > \
                   cls.max_line_length:

                # Divide along white spaces
                splitted_line = deque(line.split(" "))

                # If no split
                if splitted_line == line:
                    ret_lines.append("{0}{1}{2}".format(\
                        cls.indent*indent*cls.indent_str, line, \
                        line_ending))
                    continue

                first_line = True
                inside_str = False

                while splitted_line:
                    line_stump = []
                    indent_length = cls.indent*(indent if first_line or \
                                                 is_comment else indent + 1)
                    line_length = indent_length

                    # Check if we are not exeeding the max_line_length
                    # FIXME: Line continuation symbol is not included in
                    # FIXME: linelength
                    while splitted_line and \
                              (((line_length + len(splitted_line[0]) \
                                 + 1 + inside_str) < cls.max_line_length) \
                               or not (line_stump and line_stump[-1]) \
                               or _is_number(line_stump[-1][-1])):
                        line_stump.append(splitted_line.popleft())

                        # Add a \" char to first stub if inside str
                        if len(line_stump) == 1 and inside_str:
                            line_stump[-1] = "\""+line_stump[-1]
                            if re.search(_re_str, line_stump[-1]):
                                inside_str = not inside_str

                        # Check if we get inside or leave a str
                        if not is_comment and ("\"" in line_stump[-1] and not \
                            ("\\\"" in line_stump[-1] or \
                             "\"\"\"" in line_stump[-1] or \
                                re.search(_re_str, line_stump[-1]))):
                            inside_str = not inside_str

                        # Check line length
                        line_length += len(line_stump[-1]) + 1 + \
                                (is_comment and not first_line)*(len(\
                            cls.comment) + 1)

                    # If we are inside a str and at the end of line add
                    if inside_str and not is_comment:
                        line_stump[-1] = line_stump[-1]+"\""

                    # Join line stump and add indentation
                    ret_lines.append(indent_length*cls.indent_str + \
                                     (is_comment and not first_line)* \
                                     (cls.comment+" ") + " ".join(line_stump))

                    # If it is the last line stump add line ending otherwise
                    # line continuation sign
                    ret_lines[-1] = ret_lines[-1] + (not is_comment)*\
                                    (cls.line_cont if splitted_line else \
                                     line_ending)

                    first_line = False
            else:
                ret_lines.append("{0}{1}{2}".format(\
                    cls.indent*indent*cls.indent_str, line, \
                    line_ending))

        return ret_lines



class CCodeGenerator(BaseCodeGenerator):
    # Class attributes
    language = "C"
    line_ending = ";"
    closure_start = "{"
    closure_end = "}"
    line_cont = ""
    comment = "//"
    index = lambda self, x, i : "{0}[{1}]".format(x,i)
    indent = 2
    indent_str = " "
    to_code = lambda self, expr, name=None : ccode(expr, name)
    float_types = "REAL"

    def __init__(self, params):
        super(CCodeGenerator, self).__init__(params)


    @classmethod
    def wrap_body_with_function_prototype(cls, body_lines, name, args, \
                                          return_type="", comment="", const=False):
        """
        Wrap a passed body of lines with a function prototype.

        Input:
        ------
            body_lines: list
                List of strings where each string corresponds to one line of code.
            name: string
                Name of a prototype function.
            args: string
                String representing input arguments to the function.
            return_type: string, optional
                Type that is being returned by the function. If not given, then
                void is chosen.
            comment: string, optional
                Comment to the function placed above the function prototype.

        Return:
        -------
            A tuple of a function prototype and a function definition.
        """
        check_arg(body_lines, list)
        check_arg(name, str)
        check_arg(args, str)
        check_arg(return_type, str)
        check_arg(comment, (str, list))

        return_type = return_type or "void"

        definition = ["//"+"-"*76]
        prototype = []

        if comment:
            prototype.append("// {0}".format(comment))

        const = " const" if const else ""
        definition.append("{0} {1}({2}){3}".format(return_type, name, args, const))
        prototype.append(definition[-1])

        # Append body to prototyp
        definition.append(body_lines)

        return prototype, definition


    def module_code(self, flux):
        """
        Main function used to generate the whole C code for given Flux object.

        Returns:
        --------
            A tuple of C code. The first argument is C code of function pototypes
            usually placed in .h file. The second argument is C code of function
            definitions.
        """
        if not isinstance(flux, Flux):
            raise TypeError("*** Error: Expected a Flux object as an "\
                    "given argument.")

        h_code_lines, c_code_lines = self.total_code(flux)

        code_template = dedent("""\
        // Generated C code for the {0} model

        {1}
        """)

        return code_template.format(flux.name, "\n\n".join(h_code_lines)), \
                code_template.format(flux.name, "\n\n".join(c_code_lines))


    def total_code(self, flux):
        """
        Generate a 2-tuple of lists of code snipsets.
        The first list is C code functions prototypes and structures, and is placed
        in .h file. The second list is C code of definitions of declared functions.
        """

        check_arg(flux, Flux)

        c_code = []
        # h_code[0] stores code for struct creation, whereas h_code[1] stores
        # code for function headers
        h_code = ([], [])

        # for each flux object write code
        for flux_object in flux:
            # Code for structure definitions
            h_code[0].append(self.flux_struct(flux_object))

            # Code for init fluxes
            definition, prototype = self.init_parameters(flux_object)
            h_code[1].append(definition)
            c_code.append(prototype)

            # Code for flux function
            definition, prototype = self.flux_function(flux_object)
            h_code[1].append(definition)
            c_code.append(prototype)

            if isinstance(flux_object, FluxStochastic):

                # code for init model stochastically
                prototype, definition = self.flux_init_model_stochastically(flux_object)
                h_code[1].append(prototype)
                c_code.append(definition)

                prototype, definition = self.flux_init_model_deterministically(flux_object)
                h_code[1].append(prototype)
                c_code.append(definition)

                prototype, definition = self.flux_eval_model_stochastically(flux_object)
                h_code[1].append(prototype)
                c_code.append(definition)

                prototype, definition = self.open_states(flux_object)
                h_code[1].append(prototype)
                c_code.append(definition)

        # write code for the fluxes wrapper and append the code to the code lists
        h_code_wrap, c_code_wrap = self.construct_boundary_fluxes(flux)

        c_code.extend(c_code_wrap)
        h_code[0].extend(h_code_wrap[0])
        h_code[1].extend(h_code_wrap[1])


        return h_code[0]+h_code[1], c_code


    def get_parameters_code(self, parameters, expressions, param_name="params"):
        """
        A functions that constructs body code mapping for parameters used in
        expressions. In code generation, the order of parameters is exploited
        to define local variables with exactly the same names as parameters.
        Then the value is copied. Example of generated code:

        dt_value = params[2];
        h_value = params[4];

        Input:
        ------
            parameters: list
                A list of Params from modelparameters.
            expressions: list
                A list of sympy expressions.
            param_name: string, optional
                Refers to the way how the values of parameters are extracted.

        Return:
        -------
            A tuple of C code and a set of names of local variables.
        """
        if not isinstance(parameters, list):
            raise TypeError("*** Error: Expected list as 'parameters'.")
        if not isinstance(expressions, list):
            raise TypeError("*** Error: Expected list as 'expressions'.")
        if not isinstance(param_name, str):
            raise TypeError("*** Error: Expected string as 'param_name'.")

        # set of used parameters
        tmp_var = set()
        body_code = []

        body_code.append("//Get parameters")
        for par_ind, par in enumerate(parameters):
            # iterate over all expressions to find out if certain parameter
            # has been used
            for expr in expressions:
                try:
                    atoms = expr.expr.atoms()
                except:
                    atoms = [expr.get_sym()]

                if par.get_sym() in atoms:
                    tmp_var.add(par.name)
                    body_code.append("{0} = {1}".format(par.name, self.index(
                        param_name, par_ind)))
                    break

        body_code.append("")

        return body_code, tmp_var


    def construct_boundary_fluxes(self, flux):
        '''\
        The main function that constructs the wrapper code for flux structures
        called BoundaryFluxes. It provides both C code of a structure together
        with appriopriate functions bound to the wrapper.

        Returns:
        --------
            A tuple of C code (prototypes and definitions)
        '''

        c_code = []
        h_code = ([], [])

        # Add the wrapper struct definition
        h_code[0].append(self.wrapper_flux_struct(flux))

        # Add wrapper constructor
        h_tmp, c_tmp = self.wrapper_construct(flux)
        h_code[1].append(h_tmp)
        c_code.append(c_tmp)

        # Add wrapper initialization function
        h_tmp, c_tmp = self.wrapper_init_stochastic_boundaries(flux)
        h_code[1].append(h_tmp)
        c_code.append(c_tmp)

        # Add wrapper output
        h_tmp, c_tmp = self.wrapper_output_data(flux)
        h_code[1].append(h_tmp)
        c_code.append(c_tmp)

        # Add the wrapper mapping from boundary index into boundary flux info
        h_tmp, c_tmp = self.wrapper_flux_info()
        h_code[1].append(h_tmp)
        c_code.append(c_tmp)

        # Add the wrapper destructor
        h_tmp, c_tmp = self.wrapper_destroy(flux)
        h_code[1].append(h_tmp)
        c_code.append(c_tmp)

        return h_code, c_code


    def wrapper_flux_info(self):
        """\
        Construct C code mapping that maps boundary index into the flux information
        needed in the computations.

        Returns:
        --------
            A tuple of a function protytype and a function definition.
        """

        fluxes = self.params.code.boundary_flux.var_name
        internal = "internal_id"

        body_code = ["domain_id %s" % internal]
        body_code.append("for({2}=0; {2}<{0}->{1}; {2}++)".format(fluxes,
                         self.params.code.boundary_flux.num_of_fluxes, internal))

        body_code.append(["if({0}->{1}[{2}] == boundary_ind)".format(fluxes,
                          self.params.code.boundary_flux.boundary_map,
                          internal),
                          ["return {0}->{1}[{2}]".format(fluxes,
                           self.params.code.boundary_flux.map, internal)]])

        body_code.append("")
        body_code.append("return NULL")

        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "BoundaryFluxes_get_flux_info",
            "BoundaryFluxes_t* fluxes, domain_id boundary_ind",
            "const Flux_t*", "Map provided boundary index into the flux "\
            "information structure")

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def wrapper_construct(self, flux):
        """\
        Construct a BoundaryFlux structure's constructor. Return C code that
        reads an .h5 file and sets up all parameters in flux objects.

        Returns:
        --------
            A tuple of C code (prototype and definition)
        """
        # start building body code
        body_code = []
        wrapper_name = self.params.code.boundary_flux.var_name

        # Add all necessary information first
        body_code.append("Geometry_t* geom = species->geom")
        body_code.append("MPI_Comm comm = geom->mpi_info.comm")
        body_code.append("")
        body_code.append("BoundaryFluxes_t* {0} = mpi_malloc(comm, "\
            "sizeof(BoundaryFluxes_t))".format(wrapper_name))

        for flux_object in flux:
            if isinstance(flux_object, FluxStochastic):
                body_code.append("{0}->{1}.{2}.s0 = NULL".format(wrapper_name,
                     flux_object.var_name, self.params.code.flux.states))
        body_code.append("")

        # Initialize flux models
        body_code.append("// Initalize flux models with default parameters")
        for flux_object in flux:
            body_code.append("init_{0}_model(&{1}->{2})".format(flux_object.name,
                             wrapper_name, flux_object.var_name))

        # Read usage flags
        body_code.append("")
        body_code.append("// Read usage flags")
        for flux_object in flux:
            body_code.append('read_h5_attr(comm, file_id, "/", "use_{0}", '\
            '&{1}->use_{0})'.format(flux_object.var_name, wrapper_name))
        body_code.append("")

        # Set mappings pointers
        body_code.append("// Set mappings pointers to NULL")
        body_code.append("{0}->{1} = NULL".format(wrapper_name, self.params.code.boundary_flux.map))
        body_code.append("{0}->{1} = NULL".format(wrapper_name, self.params.code.boundary_flux.boundary_map))

        # Calculate number of fluxes
        body_code.append("")
        body_code.append("// Calculate the number of fluxes")
        body_code.append("{0}->{1} = 0".format(wrapper_name,
            self.params.code.boundary_flux.num_of_fluxes))
        for flux_object in flux:
            body_code.append("{0}->{2} += {0}->use_{1}".format(
                wrapper_name, flux_object.var_name, self.params.code.boundary_flux.num_of_fluxes))

        # Construct if guard that checks if there are fluxes that need to be constructed
        body_code.append("")
        body_code.append("// Return struct if no boundary fluxes")
        body_code.extend(["if (!{0}->{1})".format(wrapper_name,
            self.params.code.boundary_flux.num_of_fluxes), ["return fluxes"], ""])


        body_code.append("// Allocate memory for mappings")
        body_code.append("{0}->{1} = mpi_malloc(comm, sizeof(Flux_t*)*{0}->{2})".format(
            wrapper_name, self.params.code.boundary_flux.map, self.params.code.boundary_flux.num_of_fluxes))
        body_code.append("{0}->{1} = mpi_malloc(comm, sizeof(unsigned int)*{0}->{2})".format(
            wrapper_name, self.params.code.boundary_flux.boundary_map,
            self.params.code.boundary_flux.num_of_fluxes))

        # Construct map index
        map_ind = "map_ind"
        body_code.append("")
        body_code.append("// Define a map index")
        body_code.append("unsigned int %s = 0" % map_ind)

        # Construct fluxes
        for flux_object in flux:
            # Collect code lines for each flux object seperately
            flux_body_code = []
            body_code.append("")
            body_code.append("// Initialize {0} data".format(
                flux_object.var_name))

            body_code.append("if ({0}->use_{1})".format(wrapper_name,
                             flux_object.var_name))

            # one loop over all parameters to read their values from file
            flux_body_code.append("// Read {0} attribute from file".format(
                flux_object.name))
            for param in flux_object.parameters:
                # check if value for a given parameter should be read from the
                # file or set from inline arguments
                if isinstance(param, ArgCloseParam):
                    flux_body_code.append('{0}->{1}.{2} = arguments_get_t_close(arguments, "{1}")'.format(\
                        self.params.code.boundary_flux.var_name,
                        flux_object.var_name, param.name))
                else:
                    flux_body_code.append('read_h5_attr(comm, file_id, "/{1}", '\
                        '"{2}", &{0}->{1}.{2})'.format(wrapper_name,
                        flux_object.var_name, param.name))


            # Add code for reading boundary name on which the flux exsists on and
            # the species name
            flux_body_code.append("")
            flux_body_code.append('read_h5_attr(comm, file_id, "/{1}", '\
                '"boundary", &{0}->{1}.{2})'.format(wrapper_name,
                flux_object.var_name, self.params.code.flux.boundary))
            flux_body_code.append('read_h5_attr(comm, file_id, "/{1}", '\
                '"species", &{0}->{1}.{2})'.format(wrapper_name,
                flux_object.var_name, self.params.code.flux.species))

            # one loop over all parameters save them in a param array
            flux_body_code.append("")
            flux_body_code.append("// Update flux parameters")
            for ind, param in enumerate(flux_object.parameters):
                flux_body_code.append("{0}->{1}.params[{3}] = {0}->{1}.{2}".format(
                    wrapper_name, flux_object.var_name, param.name, ind))

            # construct mappings
            flux_body_code.append("")
            flux_body_code.append("// Update flux mapping")

            # construct boundary mapping
            flux_body_code.append("{0}->{3}[{2}] = Geometry_get_boundary_id(geom, "\
                "{0}->{1}.{4})".format(wrapper_name, flux_object.var_name,
                map_ind, self.params.code.boundary_flux.boundary_map,
                self.params.code.flux.boundary))

            # construct main mapping
            map_template = "{0}->{1}[{2}]".format(wrapper_name,
                self.params.code.boundary_flux.map, map_ind)

            flux_body_code.append(map_template+" = mpi_malloc(comm, sizeof(Flux_t))")
            flux_body_code.append(map_template+"->flux_params = {0}->{1}.{2}".format(
                wrapper_name, flux_object.var_name, self.params.code.flux.params))
            flux_body_code.append(map_template+"->flux_function = &{0}_flux".format(
                flux_object.var_name))
            flux_body_code.append(map_template+"->flux_dsi = "\
                "Species_get_diffusive_species_id(species, {0}->{1}.{2})".format(
                wrapper_name, flux_object.var_name, self.params.code.flux.species))
            flux_body_code.append("")

            # add states information into the mapping
            map_template += "->storef"
            if isinstance(flux_object, FluxStochastic):
                flux_body_code.append(map_template+" = mpi_malloc(comm, "\
                    "sizeof(StochasticReferences))")
                flux_body_code.append(map_template+"->states = &{0}->{1}.{2}".format(
                    wrapper_name, flux_object.var_name,
                    self.params.code.flux.states))
                flux_body_code.append(map_template+"->evaluate = "\
                    "&evaluate_{0}_stochastically".format(flux_object.name))
                flux_body_code.append(map_template+"->open_states = "\
                    "&open_states_{0}".format(flux_object.name))
            else:
                flux_body_code.append(map_template+" = NULL")

            # increase map index
            flux_body_code.append("")
            flux_body_code.append("// Increase index of used fluxes")
            flux_body_code.append("map_ind++")

            #Allocate states if needed for certain flux object
            if isinstance(flux_object, FluxStochastic):
                flux_body_code.append("")
                flux_body_code.append("// Allocate states only on rank 0 process")
                flux_body_code.append("if (geom->mpi_info.rank == 0)")
                flux_body_code.append(["// Allocate global %s states" % flux_object.name,
                    '{0}->{1}.{2}.{3} = geom->boundary_size[Geometry_get_boundary_id(geom, '\
                    '"{1}")]'.format(wrapper_name, flux_object.var_name,
                                    self.params.code.flux.states, flux_object.states[0]),
                    '{0}->{1}.{2}.{4} = mpi_malloc(comm, sizeof(domain_id)*{0}'\
                        '->{1}.{2}.{3})'.format(wrapper_name, flux_object.var_name,
                        self.params.code.flux.states, *flux_object.states)])

            # Append sub code
            body_code.append(flux_body_code)

        body_code.append("")
        body_code.append("// Sanity check")
        body_code.append("assert({0} == {1}->{2})".format(map_ind, wrapper_name,
            self.params.code.boundary_flux.num_of_fluxes))

        body_code.append("")
        body_code.append("return {0}".format(wrapper_name))

        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "BoundaryFluxes_construct", "Species_t* species, "+\
            "hid_t file_id, arguments_t* arguments", "BoundaryFluxes_t*",
            "Construct a BoundaryFlux struct")

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def wrapper_init_stochastic_boundaries(self, flux):
        """\
        Construct a C code function that initialize stochastic boundaries either
        stochastically or deterministically.

        Return:
        -------
            A tuple of C code (prototype and definition)
        """
        # Refers to information structure from arguments in fvm_mpi software
        info = "info"

        # Start buildding body code
        body_code = []
        body_code.append("unsigned int dbi")
        body_code.append("Geometry_t* geom = species->geom")
        body_code.append("BoundaryFluxes_t* fluxes = species->boundary_fluxes")
        body_code.append("const OS_info* info")
        body_code.append("")

        body_code.append("// Communicate species values at discrete "\
                "boundaries to processor 0")
        body_code.append("Species_communicate_values_at_discrete_boundaries(species)")
        body_code.append("")

        for flux_id, flux_object in enumerate(flux):
            # Skip the 'normal' fluxes, and process only those that need stochastic
            # initialization
            if not isinstance(flux_object, FluxStochastic):
                continue

            body_code.append("// Initialize {0} data".format(flux_object.var_name))
            body_code.append("if (fluxes->use_{0} && "\
                    "geom->num_global_discrete_boundaries>0)".format(flux_object.var_name))

            # Retrieve information structure
            state_code = []
            state_code.append("// Initiate state values")
            state_code.append("{0} = arguments_get_open_states_info(arguments, "\
                "fluxes->{1}.{2})".format(info, flux_object.var_name,
                self.params.code.flux.boundary))
            state_code.append("")

            # Write the code for state initialization
            state_code.append("if ({0})".format(info))
            state_code.append(["init_{0}{1}_states_deterministically(&fluxes->{2}, "\
                    "{3}->number_open_states, {3}->open_states)".format(
                    flux_object.name, self.params.code.flux.suffix,
                    flux_object.var_name, info), ""])

            state_code.append("else")
            state_code.append(["dbi = Geometry_get_discrete_boundary_id(geom, "\
                    "fluxes->{0}[{1}])".format(self.params.code.boundary_flux.boundary_map,
                    flux_id), "init_{0}{1}_states_stochastically(&fluxes->{2}, "\
                    "geom->species_values_at_local_discrete_boundaries_correct_order_rank_0[dbi])".format(
                    flux_object.name, self.params.code.flux.suffix, flux_object.var_name)])

            body_code.append(["if (geom->mpi_info.rank == 0)", state_code])
            body_code.append("")

        #write the communication code
        body_code.append("// Communicate the openness of the discrete "\
                "boundaries from rank 0 to other processes")
        body_code.append("Species_communicate_openness_of_discrete_boundaries(species)")


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "BoundaryFluxes_init_stochastic_boundaries",
            "Species_t* species, arguments_t* arguments", "void",
            "Initialize states")

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def wrapper_output_data(self, flux):
        """\
        Construct C code function that writes information about fluxes on the screen.

        Return:
        -------
            A tuple of C code (prototype and definition)
        """
        num_use_flux = "num_of_used_fluxes"
        mpi_print = "mpi_printf0(comm, "
        repeat = 77;
        body_code = []

        # start building body code
        # declare variables first
        body_code.append("BoundaryFluxes_t* fluxes = species->boundary_fluxes")
        body_code.append("Geometry_t* geom = species->geom")
        body_code.append("MPI_Comm comm = geom->mpi_info.comm")
        body_code.append("domain_id {0}".format(num_use_flux))
        body_code.append("")

        # print guard checking.
        use_flux = []
        for flux_object in flux:
            use_flux.append("fluxes->use_{0}".format(flux_object.var_name))

        body_code.append("{0} = {1}".format(num_use_flux, ' + '.join(use_flux)))
        body_code.extend(["if (!{0})".format(num_use_flux), ["return"], ""])

        # start generating outputs
        body_code.append(mpi_print+r'"\n")')
        body_code.append(mpi_print+r'"Boundary fluxes:\n")')
        body_code.append(mpi_print+r'"{0}\n")'.format('-'*repeat))
        body_code.append("")


        for flux_object in flux:
            flux_data_output = []
            flux_data_output.append(mpi_print+r'"  {0}:\n")'.format(flux_object.name))

            # Retrieve a parameter with the longest name, and tak its length
            longest_param = max(flux_object.parameters, key=lambda arg: len(arg.name))
            p_max_len = len(longest_param.name)
            for param in flux_object.parameters:
                flux_data_output.append(mpi_print+r'"  {0:>{2}}: %.5g\n", '\
                    'fluxes->{1}.{0})'.format(param.name, flux_object.var_name, p_max_len))


            stochastic_code = []
            if isinstance(flux_object, FluxStochastic):
                flux_data_output.append("")
                flux_data_output.append("if(geom->mpi_info.rank == 0)")
                stochastic_code.append("int i, open = 0")
                stochastic_code.append("for (i=0; i<fluxes->{0}.{1}.N; i++)".format(
                    flux_object.var_name, self.params.code.flux.states))
                stochastic_code.append(["open += open_states_{0}(&fluxes->{1}.{2}, i)".format(
                    flux_object.name, flux_object.var_name, self.params.code.flux.states)])

                stochastic_code.append("")
                stochastic_code.append(mpi_print+r'"  Num open {0}s: %d/%d\n", '\
                    'open, fluxes->{1}.{2}.N)'.format(flux_object.name,\
                    flux_object.var_name, self.params.code.flux.states))

                flux_data_output.append(stochastic_code)
                flux_data_output.append(mpi_print+r'"\n")')

            # append the created code
            body_code.append("if (fluxes->use_{0})".format(flux_object.var_name))
            body_code.append(flux_data_output)
            body_code.append("")


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "BoundaryFluxes_output_init_data",
            "Species_t* species, arguments_t* arguments", "void",
            "Output initial data of BoundaryFluxes")


        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def wrapper_destroy(self, flux):
        """\
        Construct C code of BoundaryFlux destructor. It frees the allocated memory
        during flux wrapper construction.

        Return:
        -------
            A tuple of C code (prototype and definition)
        """
        wrapper_name = self.params.code.boundary_flux.var_name

        body_code = ["if (!{0}->{1})".format(wrapper_name,
                     self.params.code.boundary_flux.num_of_fluxes), ["return"]]

        # add destruction of mappings
        internal = "internal_id"
        body_code.extend(["", "domain_id %s" % internal])
        body_code.append("for({2}=0; {2}<{0}->{1}; {2}++)".format(
            self.params.code.boundary_flux.var_name,
            self.params.code.boundary_flux.num_of_fluxes, internal))

        fluxes_template = "{0}->{1}[{2}]".format(self.params.code.boundary_flux.var_name,
            self.params.code.boundary_flux.map, internal)

        body_code.append(["if ({0}->storef)".format(fluxes_template),
                          ["free({0}->storef)".format(fluxes_template)],
                          "free({0})".format(fluxes_template)])
        body_code.append("free({0}->{1})".format(wrapper_name, self.params.code.boundary_flux.map))
        body_code.append("free({0}->{1})".format(wrapper_name, self.params.code.boundary_flux.boundary_map))


        # Add destruction of fluxes wrapper itself
        body_code.append("")

        for flux_object in flux:
            if isinstance(flux_object, FluxStochastic):
                body_code.append("free({0}->{1}.{2}.s0)".format(self.params.code.boundary_flux.var_name,
                                 flux_object.var_name, self.params.code.flux.states))
        body_code.append("free({0})".format(wrapper_name))


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "BoundaryFluxes_destruct", "BoundaryFluxes_t* {0}".format(\
            wrapper_name), "void", "Destroy BoundaryFlux struct")

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def wrapper_flux_struct(self, flux, comment=""):
        """
        Construct C code representation of BoundaryFlux structure.

        Returns
        -------
            String of C code
        """

        struct_code = """\
        // Boundary flux struct
        typedef struct BoundaryFluxes {{

        {0}

        }} BoundaryFluxes_t;"""

        flux_names = []
        # start building body code
        body_code = []

        body_code.append("// Flags for using ryr and serca in the model")
        body_code.append("domain_id ")
        body_code.append("")

        # add fluxes structs into the BoundaryFlux wrapper
        for flux_object in flux:
            body_code.append("// The {0} model".format(flux_object.name))
            body_code.append("{0}{1} {2}".format(flux_object.name,
                self.params.code.flux.typedef_suffix, flux_object.var_name))
            body_code.append("")

            flux_names.append(flux_object.var_name)

        # append "use_" to each element in flux_names
        flux_names = ["{0}{1}".format("use_", name) for name in flux_names]
        body_code[1] += ", ".join(flux_names)

        # Add other flux-independent variables
        body_code.append("// The number of used boundary fluxes.")
        body_code.append("domain_id num_of_used_fluxes")
        body_code.append("")

        body_code.append("// An array mapping boundary index into Flux_t structure pointer.")
        body_code.append("// The length of that map is equal to the number of used fluxes.")
        body_code.append("Flux_t** {0}".format(self.params.code.boundary_flux.map))
        body_code.append("")

        body_code.append("// An array of all boundary indices on which fluxes are applied.")
        body_code.append("unsigned int* {0}".format(
            self.params.code.boundary_flux.boundary_map))

        return dedent(struct_code).format( \
                "\n".join(self.indent_and_split_lines(body_code, indent=1)))


    def flux_struct(self, flux_object):
        '''\
        Construct a structure for certain flux with its parameters, states and
        additional arguments.

        Return:
        -------
            C code of the flux representation as a C struct.
        '''
        if not isinstance(flux_object, BaseFluxObject):
            raise TypeError("*** Error: Expects argument to be of type FluxObject.")

        # Definition of a struct model
        struct_code = """\
        // Struct keeping parameters for %s flux
        typedef struct {0} {{

        {2}

        }} {1};""" % flux_object.name

        # start building body code
        body_code = []

        # produce code lines for states if such provided
        if isinstance(flux_object, FluxStochastic):
            body_code.append("// States")
            body_code.append("ModelStates states")

#            body_code.append("unsigned int %s" % flux_object.states[0])
#            body_code.append("domain_id* %s" % flux_object.states[1])
            body_code.append("")

        # produce code lines for parameters
        body_code.append("// Parameters")
        for field in flux_object.parameters:
            body_code.append("{0} {1}".format(self.float_types, field.name))

        # add two parameters that are frequentyl used in the code
        # first add boundary name on which the flux can be computed ans the
        # species name
        body_code.extend(["", "// The name of the boundary this flux exists on"])
        body_code.append("char {0}[{1}]".format(self.params.code.flux.boundary,
            self.params.code.string_length))

        body_code.extend(["", "// The name of the species this flux is applied to"])
        body_code.append("char {0}[{1}]".format(self.params.code.flux.species,
            self.params.code.string_length))

        # secend, add temporary parameters
        body_code.extend(["", "// Temporary parameters"])
        body_code.append("{0} {1}[{2}]".format(self.float_types,
            self.params.code.flux.params, len(flux_object.parameters)))


        return dedent(struct_code).format(flux_object.name+self.params.code.flux.suffix,
            flux_object.name+self.params.code.flux.typedef_suffix,
            "\n".join(self.indent_and_split_lines(body_code, indent=1)))

    def init_parameters(self, flux_object):
        '''\
        Construct a constructor for the flux structure. The function initializes
        parameters using values from .flux file

        Return:
        -------
            A tuple of C code. The first argument is a function definition only,
            whereas the second argument is the main body of the function wrapped
            within its definition.
        '''
        # start building body code
        body_code = []

        access_template = "{0}->".format(flux_object.var_name)
        for num, field in enumerate(flux_object.parameters):
            body_code.append("{0}{1} = {2}".format(access_template, field.name, field.value))
            body_code.append("{0}{1} = {0}{2}".format(access_template,
                self.index(self.params.code.flux.params, num), field.name))

            if num < len(flux_object.parameters)-1:
                body_code.append("")

        # Add function prototype
        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "init_{0}_model".format(flux_object.name),
            "{0}_model_t* {1}".format(flux_object.name, flux_object.var_name),
            "void", comment="Initialize a {0}_model".format(flux_object.name))

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def flux_function(self, flux_object):
        '''\
        Construct C code of the flux function for certain flux type.

        Return:
        -------
            A tuple of C code. The first argument of the tuple is a function definition
            (which is placed in .h file), whereas the second argument is the function
            code to be placed in .c file
        '''
        # define temporary variables used in calculations
        # start building body code and reserve one line for declarations
        body_code, tmp_var = self.get_parameters_code(flux_object.parameters,
                flux_object.expressions, self.params.code.flux.params)

        # Calculate flux from expressions
        body_code.append("//Main calculations")
        for expr in flux_object.expressions[:-1]:
            tmp_var.add(expr.name)
            body_code.append(self.to_code(expr.expr, expr.name))

        body_code.insert(0, "")
        body_code.insert(0, "{0} {1}".format(self.float_types, ', '.join(tmp_var)))
        body_code.append("")

        # Add return value
        body_code.append("return {0}".format(self.to_code(flux_object.flux_value.expr)))

        # Add function prototype
        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "{0}_flux".format(flux_object.var_name),
            ", ".join(["{0} {1}".format(self.float_types, var_name) \
                for var_name in flux_object.variables]+["{0}* {1}".format(self.float_types,
                    self.params.code.flux.params)]),
            self.float_types)


        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def flux_init_model_stochastically(self, flux_object):
        """\
        Construct C code of the init function for certain flux that requires
        stochastic calculations.

        Return:
        -------
            A tuple of C code. The first argument of the tuple is a functio
            definition, whereas the second one is the function prototype.
        """

        submodel = flux_object.submodel
        rand_var = "_r_var"

        # start building body code
        body_code, tmp_var = self.get_parameters_code(flux_object.parameters,
                submodel.expressions, "{0}->{1}".format(flux_object.var_name,
                self.params.code.flux.params))

        body_code.extend(["if ({0}->{1}.N == 0)".format(flux_object.var_name,
                          self.params.code.flux.states), ["return"]])
        body_code.append("")

        body_code.append("// Assume that the state arrays have been constructed")
        body_code.append("assert({0}->{1}.s0)".format(flux_object.var_name,
                         self.params.code.flux.states))
        body_code.append("")

        body_code.append("for ({0}=0; {0}<{1}->{2}.N; {0}++)".format(submodel.i,
                         flux_object.var_name, self.params.code.flux.states))

        for_body_loop = []

        # add all code line that symbols occure in a set of expressions
        for expr in submodel.expressions:
            tmp_var.add(expr.name)
            for_body_loop.append(self.to_code(expr.expr, expr.name))
        for_body_loop.append("")

        # add code for seting the state value
        tmp_var.add(rand_var)
        pr = submodel.rates.probabilities()
        total_possible_values = len(pr.keys())

        # get a random number
        for_body_loop.append("%s = %s" % (rand_var, self.params.symbols.variables.rand))
        # Find the state value

        for nr, (key, value) in enumerate(pr.iteritems(), 1):
            if nr == 1:
                total_value = value
                if_statement = "if ({0} <= {1})".format(rand_var, self.to_code(total_value))
            elif nr == total_possible_values:
                if_statement = "else"
            else:
                total_value += value
                if_statement = "else if ({0} <= {1})".format(rand_var, self.to_code(total_value))

            for_body_loop.extend([if_statement, ["{0}->{1}.s0[{2}] = {3}".format(
                                  flux_object.var_name, self.params.code.flux.states,
                                  submodel.i, self.to_code(key))]])

        body_code.append(for_body_loop)

        body_code.insert(0, "")
        body_code.insert(0, "{0} {1}".format(self.float_types, ', '.join(tmp_var)))
        body_code.insert(0, "unsigned int {0}".format(submodel.i))


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "init_{0}{1}_states_stochastically".format(flux_object.name,\
                self.params.code.flux.suffix),  "{0}{1} *{2}, ".format(flux_object.name,\
                self.params.code.flux.typedef_suffix, flux_object.var_name) + \
                "{0}* {1}".format(self.float_types, submodel.species), "void",
                "Initialize the {0} states stochastically".format(flux_object.name))


        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def flux_init_model_deterministically(self, flux_object):
        """
        Construct C code function that initialize states' values deterministically.

        Return:
        -------
            A tuple of C code (prototype and definition)
        """

        open_ = self.params.symbols.variables.open+flux_object.var_name
        init_ = self.params.symbols.variables.init+flux_object.var_name

        # start building body code
        body_code = []
        body_code.extend(["if ({0}->{1}.N == 0)".format(flux_object.var_name,
                          self.params.code.flux.states), ["return"]])
        body_code.append("")

        idx = "i"
        body_code.append("unsigned int {0}".format(idx))
        body_code.append("")

        body_code.append("// Assume that the state array have been constructed")
        body_code.append("assert({0}->{1}.s0)".format(flux_object.var_name,
                         self.params.code.flux.states))
        body_code.append("")

        # set values of all states to 0
        body_code.append("for ({0}=0; {0}<{1}->{2}.N; {0}++)".format(idx,
                         flux_object.var_name, self.params.code.flux.states))
        body_code.append(["{0}->{1}.s0[{2}] = 0".format(flux_object.var_name,
                          self.params.code.flux.states, idx)])
        body_code.append("")

        body_code.append("for ({0}=0; {0}<{1}; {0}++)".format(idx, init_))
        body_code.append(["{0}->{1}.s0[{2}[{3}]] = 1".format(flux_object.var_name,
                          self.params.code.flux.states, open_, idx)])

        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "init_{0}{1}_states_deterministically".format(flux_object.name,\
                self.params.code.flux.suffix), "{0}{1} *{2}, ".format(flux_object.name,\
                self.params.code.flux.typedef_suffix, flux_object.var_name) +
                "unsigned int {0}, ".format(init_) + "int* {0}".format(open_), "void",
                "Initialize the {0} states deterministically".format(flux_object.name))

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))


    def flux_eval_model_stochastically(self, flux_object):
        """
        Construct C code function that evaluates states transitions stochastically.

        Return:
        -------
            A tuple of C code (prototype and definition)
        """

        submodel = flux_object.submodel
        rand_var = "_r_var"

        # start building body code
        body_code, tmp_var = self.get_parameters_code(flux_object.parameters,
                submodel.expressions+[flux_object.close], self.params.code.flux.params)

        body_code.extend(["if ({0}->N == 0)".format(self.params.code.flux.states),
                          ["return"]])
        body_code.append("")

        body_code.append("// Assume the state arrays has been constructed")
        body_code.append("assert({0}->s0)".format(self.params.code.flux.states))
        body_code.append("")

        body_code.append("for ({0}=0; {0}<{1}->N; {0}++)".format(submodel.i,
                         self.params.code.flux.states))

        states_template = "{0}->s0[{1}]".format(self.params.code.flux.states, submodel.i)
        for_body_loop = []
        # add check statement for forcing flux to be closed or open
        for_body_loop.extend(["//Check for forcing flux to be closed or open",
                              "if (%s>0)" % flux_object.close.name,
                              ["//Check for close time",
                              "if (%s <= %s+t)" % (flux_object.close.name,
                                                  self.params.symbols.variables.dt),
                              ["{0} = {1}".format(states_template, 0)], "",
                              "// Do not continue to stochastic evaluation",
                              "continue"]])
        for_body_loop.append("")

        # add all code line that symbols occure in a set of expressions
        for expr in submodel.expressions:
            tmp_var.add(expr.name)
            for_body_loop.append(self.to_code(expr.expr, expr.name))
        for_body_loop.append("")

        # Go through all possible states values and compute possible transition
        # to another state
        tmp_var.add(rand_var)

        for_body_loop.append("{0} = {1}".format(rand_var, self.params.symbols.variables.rand))
        switch_code = []

        for from_state in submodel.rates.states():
            transition_states = submodel.rates.transition(from_state, self.params.symbols.variables.dt)

            case_code = []
            for ind, to_state in enumerate(transition_states, 1):
                if ind == 1:
                    total_value = to_state[1]
                    if_statement = "if ({0} <= {1})".format(rand_var, self.to_code(total_value))
                else:
                    total_value += to_state[1]
                    if_statement = "else if ({0} <= {1})".format(rand_var, self.to_code(total_value))

                case_code.extend([if_statement, ["{0} = {1}".format(states_template,
                                                 to_state[0])]])

            case_code.append("break")
            switch_code.extend(["case {0}:".format(from_state), case_code])

#        switch_code.extend(["default:", ['mpi_printf_error(comm, "*** ERROR: Invalid '\
#                'state encountered. Got %d.", {0})'.format(states_template)]])


        for_body_loop.extend(["switch ({0})".format(states_template), switch_code])
        body_code.append(for_body_loop)

        body_code.insert(0, "")
        body_code.insert(0, "{0} {1}".format(self.float_types, ', '.join(tmp_var)))
        body_code.insert(0, "unsigned int {0}".format(submodel.i))


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "evaluate_{0}_stochastically".format(flux_object.name),
                "{0} *{1}, ".format(self.float_types, self.params.code.flux.params)\
                + "{0}* {1}, ".format("ModelStates", self.params.code.flux.states)\
                + "{0} t, ".format(self.float_types) \
                + "{0} {1}, ".format(self.float_types, self.params.symbols.variables.dt)\
                + "{0}* {1}".format(self.float_types, self.params.symbols.variables.species),
                "void", "Evaluate the {0} states stochastically".format(flux_object.name))


        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))

    def open_states(self, flux_object):
        assert(isinstance(flux_object, FluxStochastic))

        submodel = flux_object.submodel
        states = self.params.code.flux.states

        body_code = []
        body_code.append("assert({0} < {1}->N)".format(submodel.i, states))

        # Get all states that are to be considered to be open
        open_ = sorted(submodel.states_values.open())
        if len(open_) == 0:
            raise RuntimeError("*** Error: No open states porvided for %s flux. "\
                               "Check if open_states are defined in the "\
                               "corresponding .mm file." % flux_object.name)
        open_no = len(open_)

        # If we have more than one state being considered to be open, then
        # generate a for-loop
        if open_no > 1:
            body_code.append("")
            body_code.append("// Define \"open\" states")
            body_code.append("unsigned int j, open[%d] = {%s}" % (open_no,
                                                   ','.join(map(str, open_))))
            body_code.append("for(j=0; j<{0}; j++)".format(open_no))
            body_code.append(["if ({0}->s0[{1}] == open[j])".format(states,
                              submodel.i), ["return 1"]])
            body_code.append("return 0")
        else:
            body_code.append("return {0}->s0[{1}] == {2}".format(states,
                             submodel.i, open_[0]))


        fun_prototype, fun_definition = self.wrap_body_with_function_prototype(\
            body_code, "open_states_{0}".format(flux_object.name),
            "const ModelStates* states, int {}".format(submodel.i), "domain_id",
            "Return 1 if the i-th flux is open. Otherwise 0.")

        return "\n".join(self.indent_and_split_lines(fun_prototype)), \
                "\n".join(self.indent_and_split_lines(fun_definition))