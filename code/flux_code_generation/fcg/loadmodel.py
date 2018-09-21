#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models import Flux, MarkovModel
from base import *

import os
import weakref


class IntermediateDispatcher(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self._model = None
    
    def __setitem__(self, name, value):
        self.model.register_expression(name, value)
        
    def __getitem__(self, name):
        # first check if the name is in a global namespace
        try:
            return super(IntermediateDispatcher, self).__getitem__(name)
        except KeyError:
            # if it is not, check the flux object local namespace
            return self.model.get_symbol(name)

            
    @property
    def model(self):
        if self._model is None:
            raise ValueError('*** Error: No model object has been stored.')
        return self._model()

        
    @model.setter
    def model(self, model):
        if not isinstance(model, AbstractBaseObject):
            raise TypeError("*** Error: Expected a object being a subclass of "\
                "BaseInterface. Instead got {0}".format(type(model)))
        self._model = weakref.ref(model)


def load_model(filename, params=None):
    return _load_model(filename, Flux, params=params)


def _load_model(filename, ModelType, **kwargs):
    if not isinstance(filename, str):
        raise TypeError("*** Error: Expected a string as 'filename'.")
    if not issubclass(ModelType, AbstractBaseObject):
        raise TypeError("*** Error: Expected a subclass of BaseModel. Instead "\
            "obtained an object of type {0}.".format(type(ModelType)))
    
    # create a model
    model = ModelType(filename, **kwargs)
    
    # create a namespace object that will collect variables from a file
    intermediate_namespace = IntermediateDispatcher()
    intermediate_namespace.model = model
    
    # fill out namespace with relevant object to be able to parse the file    
    _init_model_namespace(intermediate_namespace)
    
    if not os.path.isfile(filename):
        raise ValueError('*** Error: Could not find {0}'.format(filename))
        
    execfile(filename, intermediate_namespace)
    
    return model


def _init_model_namespace(namespace):
    """
    Register function namespace used in .flux files
    """
   
    def flux(name, external_file=None, **kwargs):
        if external_file is None:
            namespace.model.add_flux_object(name, FluxNormal, kwargs)
        else:
            namespace.model.add_flux_object(name, FluxStochastic, kwargs)
    
            # when a special flux has been encoutered, read and additional file
            submodel = _load_model(os.path.join(namespace.model.path, external_file), 
                                   MarkovModel,
                                   arguments=namespace.model.current_flux.get_parameters_symbols(),
                                   params=namespace.model.get_params())
            
            namespace.model.current_flux.submodel = submodel
        
    # update namespace with the above defined functions    
    namespace.update(dict(Flux=flux))
