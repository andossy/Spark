#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import weakref
import sympy as sp

from base import *
from sympytools import symbols_from_expr

class Flux(AbstractBaseObject):
    """
    Main class considered as a contianer to store all flux objects. It provides
    detailed interface for registering expressions and collecting symbols in/from
    current flux object.
    """
    def __init__(self, filename, **kwargs):
        if not isinstance(filename, str):
            raise TypeError("***Error: Expected a string argument.")

        self._path, name = os.path.split(filename)
        name = os.path.splitext(name)[0]

        super(Flux, self).__init__(name)

        # store a main file name
        self._filename = filename

        # store parameters
        self._params = kwargs['params']

        # store a list of FluxObjects
        self._flux_objects = []
        self._current_flux = None


    def add_flux_object(self, name, FluxObjectRef, kwargs):
        """
        Add a new flux object to the container.

        Input:
        ------
            name: str
                name of the newly created flux object
            FluxObjectRef: class
                a detailed type of the created flux object
            kwargs: dict
                a map with defined flux parameters together with their values
        """
        self._flux_objects.append(FluxObjectRef(name, kwargs))
        self._current_flux = weakref.ref(self._flux_objects[-1])

        # update local namespace with local variables used in flux calculations
        self.current_flux._add_flux_variables([self._params.variables.dt,
                                              self._params.variables.h,
                                              self._params.variables.loc_one,
                                              self._params.variables.loc_two])

        if isinstance(self.current_flux, FluxStochastic):
            self.current_flux._add_flux_states(self._params.states.length,
                                               self._params.states.variable)

    @property
    def current_flux(self):
        """
        Return current (being considered) flux object. Useful in code generation.
        """
        return self._current_flux()


    @property
    def filename(self):
        """
        Return a name of currently processed file.
        """
        return self._filename

    @property
    def path(self):
        """
        Return a path  to the currently processed file.
        """
        return self._path


    def get_params(self):
        """
        Return parameters object.
        """
        return self._params


    def __iter__(self):
        """
        Return iterator to flux objects.
        """
        return self._flux_objects.__iter__()


    def get_fluxes_names(self):
        """
        Return a list of flux object names.
        """
        names = []
        for flux in self._flux_objects:
            names.append(flux.name)

        return names


    def get_flux_object(self, name):
        """
        Return appriopriate flux object that name corresponds with 'name'.

        Input:
        ------
            name: str
                name of flux object

        Return:
        -------
            flux object: FluxObject
        """
        if not isinstance(name, str):
            raise TypeError("*** Error: Expects 'name' to be a string.")

        for flux in self._flux_objects:
            if flux.name == name:
                return flux

        raise ValueError("*** Error: There is not flux object with an assosiate \
                         name: {0}.".format(name))


    def get_symbol(self, name):
        return self.current_flux.get_symbol(name)


    def register_expression(self, variable, expression):
        self.current_flux.register_expression(variable, expression)



class MarkovModel(AbstractObject):
    """
    A class used to define a Markov model being a submodel of certain
    StochasticFlux object. It keeps track of all mathematical expressions
    appearing in an additional .mm file, and rates functions used in state changes.
    """

    def __init__(self, name, **kwargs):
        super(MarkovModel, self).__init__(name)
        self.rates = Rates()
        self.states_values = StatesValues()
        # no longe needed
        self._eval = dict()

        # set symbols to current namespace
        for symbol in kwargs['arguments']:
            self._namespace[symbol.name] = symbol

        symbols = kwargs['params'].variables
        self._namespace[symbols.dt] = sp.Symbol(symbols.dt)
        self._namespace[symbols.h] = sp.Symbol(symbols.h)
        self._namespace[symbols.species] = sp.IndexedBase(symbols.species)
        self._namespace[symbols.iterate] = sp.Symbol(symbols.iterate)
        self._namespace["rates"] = self.rates
        self._namespace["states_values"] = self.states_values

        self._i = symbols.iterate
        self._species = symbols.species

    def set_init(self, expr):
        self._init = expr

    def set_init2(self, expr, true, false):
        self._init = (Intermediate('init', expr), true, false)

    def set_eval(self, type_, expr):
        self._eval[type_] = expr

    @property
    def i(self):
        return self._i

    @property
    def species(self):
        return self._species


class Rates(dict):
    """
    A helper class provided to store rate functions.
    """
    def __init__(self):
       super(Rates, self).__init__()
       self._probabilities = dict()
       self._total = sp.Number(0.0)

    def __setitem__(self, states, expr):
        # First check if states is a tuple of length two with interegers
        if not (isinstance(states, tuple) and len(states) == 2 and
                all(isinstance(state, int) for state in states)):
            raise RuntimeError("*** Error: Expected two integers defining a "\
                "direction of rate as key. Got %s\n" % str(states))

        expr = sp.sympify(expr, evaluate=False)
        # Next check for valid rate expression
        if not isinstance(expr, sp.Basic):
            raise TypeError("*** Error: Expects sympy expressions or numbers.")

        self._total += expr
        try:
            self._probabilities[states[0]] += expr
        except KeyError:
            self._probabilities[states[0]] = expr

        super(Rates, self).__setitem__(states, expr)

    def probabilities(self):
        """
        Function that returns a cumulative distribution of function rates.
        """
        copy = self._probabilities.copy()
        for key in copy.iterkeys():
            copy[key] /= self._total

        return copy


    def states(self):
        """
        Return a set of all possible states we can be in.
        """
        st = set()

        for states in self.iterkeys():
            st.add(states[0])
            st.add(states[1])

        return st


    def transition(self, from_state, var=None):
        """
        Return a list of possible states we can go in one step starting from
        'from_state' with the appropriate probability.
        """
        transition = []
        for states, values in self.iteritems():
            if states[1] == from_state:
                values = values if var is None else values*sp.Symbol(var)
                transition.append((states[0], values))

        return transition


class StatesValues(dict):
    """
    Stores states which are considered as open
    """
    def __setitem__(self, state, expr):
        if not isinstance(state, (int, long)):
            raise RuntimeError("*** Error: Expects an integer defining which "\
                    "state should be considered as open. By default all are "\
                    "closed.")

        if not expr in [0,1]:
            raise RuntimeError("*** Error: Only values 0 or 1 (state is open/closed "\
                    "respectively) can be set to state_vales corresposing to Got {}".format(expr))

        super(StatesValues, self).__setitem__(state, expr)

    def __getitem__(self, state):
        if not isinstance(state, (int, long)):
            raise RuntimeError("*** Error: Expects an integer defining which "\
                    "state should be considered as open. By default all are "\
                    "closed.")

        # Return a stored value. If not stored return 0
        return self.get(state, 0)

    def open(self):
        """
        Returns states which are considered to be open.
        """
        op = []
        for state, value in self.iteritems():
            if value:
                op.append(state)

        return op
