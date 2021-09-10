"""This module contains miscellaneous utilities."""

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "MIT"


def flatten(list):
    return [item for sublist in list for item in sublist]
