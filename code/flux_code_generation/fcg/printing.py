#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The module is provided since integers, using ModelParameters code generation, 
are printed in a float-like form. This may cause unwanted errors in generated
code. For instance in indexing arrays.
"""

from modelparameters.codegeneration import _CustomCCodePrinter
from distutils.version import LooseVersion as _V
import sympy as sp

_current_sympy_version = _V(sp.__version__)

# Check version for order arguments
if _current_sympy_version >= _V("0.7.2"):
    _order = "none"
else:
    _order = None


class _CCodePrinter(_CustomCCodePrinter):
    """
    Code generation class provided to overwrite functions responsible for
    printing integers. In the super class integers are printed in a float
    format which is hardly applicable in array indexing.
    """
    
    def _print_One(self, expr):
        return "1"
        
    def _print_Zero(self, expr):
        return "0"
        
    def _print_NegtiveOne(self, expr):
        return "-1"
        
    def _print_Integer(self, expr):
        return "{0}".format(expr.p)
        

_ccode_printer = _CCodePrinter(order=_order, full_prec=False)


def ccode(expr, assign_to=None):
    """
    Return a C-code representation of a sympy expression
    """
    ret = _ccode_printer.doprint(expr)
    if assign_to is None:
        return ret
    return "{0} = {1}".format(assign_to, ret)
