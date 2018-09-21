#!/usr/bin/env python
# -*- coding: utf-8 -*-

from modelparameters.sympytools import *
from modelparameters.parameters import ScalarParam

class ArgCloseParam(ScalarParam):
    '''
    This class is provided to distinguished between parameters that must be read
    from .h5 file and those that must be initialized from the command line arguments.

    This parameter will hold t_close value provided from command line.
    '''
    def __init__(self):
        ScalarParam.__init__(self, -1)


sp_namespace["ScalarParam"] = ScalarParam
sp_namespace["ArgCloseParam"] = ArgCloseParam

__all__ = [_name for _name in globals().keys() if not _name.startswith('_')]
