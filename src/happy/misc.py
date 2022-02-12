# ------------------------------------------------------------------------------
#  File: misc.py
#  Author: Jan Kukacka
#  Date: 6/2018
# ------------------------------------------------------------------------------
#  Miscelaneous functions
# ------------------------------------------------------------------------------


def ensure_list(var):
    '''
    Helping function for scenarios when variable (usually function parameter)
    is allowed to be an object or a list of objects.
    This returns canonical representation where even single objects are converted
    to a list containing a single object.

    Taken from https://stackoverflow.com/a/1416677/2042751.
    '''
    if isinstance(var, str):
        var = [var]
    ## Dictionaries are iterables but we consider them normal objects for this
    ## purpose.
    elif isinstance(var, dict):
        var = [var]
    else:
        try:
            iter(var)
        except TypeError:
            var = [var]
    return var
