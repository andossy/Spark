#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ModelParameters imports
from fcg.sympytools import sp_namespace, ArgCloseParam
from modelparameters.parameters import ScalarParam
from gotran import Intermediate

import sympy as sp
from string import lower
from abc import ABCMeta, abstractmethod

from warnings import warn


class Base(object):
    """
    A base class for all Flux objects
    """

    def __init__(self, name):
        super(Base, self).__init__()
        if not isinstance(name, str):
            raise TypeError("***Error: Expected 'name' to be a string.")
        self._name = name

    @property
    def name(self):
        """
        Returns a name of a flux object.
        """
        return self._name


class Interface(object):
    __metaclass__ = ABCMeta
    """
    A base class defining an interface providing two main functions used in
    IntermediateDispatcher to read files: get_symbol, register_expression
    """
    def __init__(self):
        super(Interface, self).__init__()

    @abstractmethod
    def get_symbol(self, name):
        """
        Return a sympy symbol for given 'name'.
        """
        return

    @abstractmethod
    def register_expression(self, variable, expression):
        """
        Register an occurred sympy expression.

        Input:
        ------
            variable: str
                A varaible name. The one the stands on the left-hand-side of '='
            expression: sympy expression or number
               A symbolic expression that stands on the right-hand-side of '='
        """
        return


class AbstractBaseObject(Base, Interface):
    """
    An abstract base class for all objects that require certain interface.
    """
    def __init__(self, name):
        super(AbstractBaseObject, self).__init__(name)


class AbstractObject(AbstractBaseObject):
    """
    An abstract object that holds special variables for sympy symbols and
    sympy expressions, and provides detailed interface to register expressions
    and to collect processed symbols.
    """
    def __init__(self, name):
        super(AbstractObject, self).__init__(name)

        # namespace container that stores sympy functions and symbols used
        # to define math expressions
        self._namespace = dict()
        self._namespace.update(sp_namespace)

        # list of all expressions in order of their apperance. The expressions
        # are stores as Intermediates from the gotran
        self._expressions = []


    def get_symbol(self, name):
        if not isinstance(name, str):
            raise TypeError("*** Error: Expected a string as a 'name'.")
        return self._namespace[name]

    def register_expression(self, variable, expression):
        expression = sp.sympify(expression, evaluate=False)

        if not isinstance(expression, sp.Basic):
            raise TypeError("*** Error: Expects sympy expressions.")

        # set up a new intermediate expression
        i_expr = Intermediate(variable, expression)

        # register an expression
        self._expressions.append(i_expr)

        # register a sympy symbol in a symbols' array
        self._namespace[i_expr.name] = i_expr.sym


    @property
    def expressions(self):
        """
        Returns a list of sympy expressions embedded in Intermediates wrappers
        from gotran library. The order of the expressions is equivalent to the
        order of their appearance in a .flux file.
        """
        return self._expressions


class BaseFluxObject(AbstractObject):
    """
    A base class for all flux structures. At the moment only two types are
    distinguished. Both inherits from this class providing neccessary local
    variables if needed.
    """

    def __init__(self, name, args):
        super(BaseFluxObject, self).__init__(name)

        # name of a variable of type flux
        self._variable_name = lower(name)

        # list of ScalarParams defining the flux structure
        self._parameters = []

        # save arguments and their values
        self._register_arguments(args)


    def _add_flux_variables(self, params):
        """
        Register local arguments that can be used in a function responsible for
        calculating fluxes. Shoud be used internally only.
        """
        if not isinstance(params, (tuple, list)):
            raise TypeError("*** Error: Expect 'params' to be a list-like object \
                            of strings.")

        self._flux_variables = params

        # register each parameter in a symbol table
        for par in params:
            self._namespace[par] = sp.Symbol(par)


    def _register_arguments(self, kwargs):
        """
        Registers arguments that occur in flux structures. The function should
        be used internally only.

        Input:
        ------
            kwargs: dict
                A dicitonary with arguments as keys and their initial values
                as values.
        """
        if not isinstance(kwargs, dict):
            raise TypeError("*** Error: 'args' must be a dictionary with keys \
                            as strings and numeric values.")

        for field_name, value in kwargs.iteritems():
            if not isinstance(field_name, str):
                raise TypeError("*** Error: Flux argument name must be a string.")
            if not isinstance(value, (float, int, ScalarParam)):
                raise TypeError("*** Error: Flux field must have a numeric value.")

            if isinstance(value, ScalarParam):
                field = value
                field.name = field_name
            else:
                field = ScalarParam(value, name=field_name)
            self._parameters.append(field)
            self._namespace[field.name] = field.get_sym()


    def register_expression(self, variable, expression):
        """
        The function check whethers the final expression has been already defined.
        If yes, then other expressions occuring after the final one are not accepted.
        """
        if len(self.expressions) > 0 and \
                self.expressions[-1].name == "d_{0}".format(self.name):
            warn("*** WARNING: The final expression 'd_{0}' has already been defined. "\
                "Therefore the expression '{1}' being processed will be omitted.\n".format(self.name, variable))
            return

        super(BaseFluxObject, self).register_expression(variable, expression)


    @property
    def var_name(self):
        """
        Returns a name of a variable of type FluxObject. In particular, it is
        just a Flux name written in lowercase letters.
        """
        return self._variable_name


    @property
    def parameters(self):
        """
        Returns a list of ScalarParams being fields in flux structure in generated
        code. These parameters are the ones defined in the Flux function in
        a .flux file.
        """
        return self._parameters


    def get_parameters_symbols(self):
        """
        Returns a list of parameter symbols without their values.
        """
        return [param.sym for param in self.parameters]


    @property
    def flux_value(self):
        """
        Returns a sympy expression that is the final evaluation of flux. This
        expression is returned in generated code.
        """
        expr = self._expressions[-1]
        if expr.name != 'd_{0}'.format(self.name):
            raise RuntimeError("*** Error: The final flux expression has not "\
                "been defined. In the list of expressions should have an "\
                "expression with name d_{0}.".format(self.name))

        return expr

    @property
    def variables(self):
        """
        Return a list of local variables used in a function that computes flux.
        By default, the list contains strings: 'dt', 'h', 'u0', 'u1'; however,
        they may differ if certain values change in passed parameters.
        """
        return self._flux_variables


class FluxNormal(BaseFluxObject):
    """
    Class for usual fluxes, i.e. the ones that do not need any stochastic
    evaluations, and therefore, they do not need additional parameters. The class
    is used only for type checking.
    """
    pass


class FluxStochastic(BaseFluxObject):
    """
    Class for fluxes that need some stochastic evaluations. The only difference
    is that such flux object should contain special state variables. It also
    to determine whether the considered flux is a stochastic one or not.
    """
    def __init__(self, name, args):
        super(FluxStochastic, self).__init__(name, args)

        self.close = []
        # Find a parameter for closing time
        for p_ind, p in enumerate(self._parameters):
            if isinstance(p, ArgCloseParam):
                p.value = p_ind
                self.close.append(p)

        # Check whether the user provided close parameter. Must be only one!
        error_cmd = "*** Error: %s flux. For stochastic fluxes exactly one "\
                    "parameter of type ArgCloseParam must be provided." % name

        num_close = len(self.close)
        if num_close == 0:
            raise RuntimeError(error_cmd)
        if num_close > 1:
            raise RuntimeError(error_cmd + " Found %d such parameters: %s" \
                              % (num_close, ', '.join(map(lambda x: x.name, self.close))))

        self.close = self.close[0]
        self._stochastic_expressions = []


    def _add_flux_states(self, var_len, var_var):
        """
        Register state variables for certain flux. This states are used in
        stochastic evaluations. The function should be used internally only.
        """
        self._states = (var_len, var_var)

    @property
    def states(self):
        """
        Return a state tuple. The first argument is the variable name for number
        of states. The second one is the variable name for state pointer.
        """
        return self._states

    @property
    def submodel(self):
        """
        Return an instance of a submodel.
        """
        return self._submodel

    @submodel.setter
    def submodel(self, submodel):
        self._submodel = submodel
