#!/usr/bin/env python3
import argparse
import os
import sys
from itertools import product

import numpy as np

from table_tools import read_footprint_file


def footprint_area(RAmin, RAmax, DECmin, DECmax):
    """
    Calculate the area within a RA/DEC bound.

    Parameters
    ----------
    RAmin : array_like or float
        Minimum right ascension of the bounds.
    RAmax : array_like or float
        Maximum right ascension of the bounds.
    DECmin : array_like or float
        Minimum declination of the bounds.
    DECmax : array_like or float
        Maximum declination of the bounds.

    Returns
    -------
    area : array_like or float
        Area within the bounds in square degrees.
    """
    # np.radians and np.degrees avoids rounding errors
    sin_DEC_min, sin_DEC_max = np.sin(np.radians([DECmin, DECmax]))
    dRA = RAmax - RAmin
    if RAmin > RAmax:
        dRA += 360.0
    area = dRA * np.degrees(sin_DEC_max - sin_DEC_min)
    return area


def next_DEC(area, RAmin, RAmax, current_DEC):
    """
    Calculate the upper declination bound of a tile based on its lower bound
    declination, extend in right ascension and tile area.

    Parameters
    ----------
    area : array_like or float
        Area the tile should cover in square degrees.
    RAmin : array_like or float
        Minimum right ascension of the tile in degrees.
    RAmax : array_like or float
        Maximum right ascension of the tile in degrees.
    current_DEC : array_like or float
        Minimum declination of the tile in degrees.

    Returns
    -------
    next_DEC : array_like or float
        Maximum declination in degrees of the tile such that it fits the
        requested area.
    """
    # np.radians and np.degrees avoids rounding errors
    area_sterad = np.radians(np.radians(area))
    dRA = RAmax - RAmin
    if RAmin > RAmax:
        dRA += 360.0
    sin_current_DEC = np.sin(np.radians(current_DEC))
    arcsin_argument = area_sterad / np.radians(dRA) + sin_current_DEC
    if arcsin_argument >= 1.0:
        next_DEC = 90.0
    elif arcsin_argument <= -1.0:
        next_DEC = -90.0
    else:
        next_DEC_rad = np.arcsin(arcsin_argument)
        next_DEC = np.degrees(next_DEC_rad)
    return next_DEC


def pointing_name(prefix, RAmin, RAmax, DECmin, DECmax):
    """
    Find an automatic name for a pointing based on its centeral coordinate.
    This matches the KiDS internal naming scheme.

    Parameters
    ----------
    RAmin : float
        Minimum right ascension of the tile in degrees.
    RAmax : float
        Maximum right ascension of the tile in degrees.
    DECmin : float
        Minimum declination of the tile in degrees.
    DECmax : float
        Maximum declination of the tile in degrees.

    Returns
    -------
    name : str
        Propsed name for the tile.
    """
    RAcenter = (RAmin + RAmax) / 2.0
    DECcenter = (DECmin + DECmax) / 2.0
    fragments = [
        prefix.strip("_"),
        ("%.1f" % RAcenter).replace(".", "p"),
        ("%.1f" % DECcenter).replace(".", "p").replace("-", "m")]
    return "_".join(fragments)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate a rectangular survey footprint from RA and DEC '
                    'bounds, represented by a STOMP map. The bounds are '
                    'appended to the file ./footprint.txt. Optionally split '
                    'the footprint into equal area pointings, exported to '
                    'a pointings list file.')
    parser.add_argument(
        '-b', '--bounds', nargs=4, type=float, required=True,
        help='bounds of polygon in degrees: RA_min RA_max DEC_min DEC_max')
    parser.add_argument(
        '-f', '--footprint-file', default="footprint.txt",
        help='file in which the survey meta data is collected '
             '(default: %(default)s)')
    parser.add_argument(
        '--survey', required=True,
        help='name to idenify survey in ./footprint.txt')
    parser.add_argument(
        '-p', '--pointings-file',
        help='file in which the pointing boundaries are collected '
             '(default: do not create this file)')
    parser.add_argument(
        '--grid', nargs=2, type=int,
        help='number of pointings along the RA and the DEC axis of the '
             'pointing grid (n_RA x n_DEC), format: n_RA n_DEC '
             '(requires --pointings-file)')
    args = parser.parse_args()

    RAmin, RAmax, DECmin, DECmax = args.bounds
    # check bounds
    if not all(-90.0 <= dec <= 90.0 for dec in (DECmin, DECmax)):
        sys.exit(
            "ERROR: DEC_min and DEC_max must be between -90 and 90 degrees")
    if not all(0.0 <= ra <= 360.0 for ra in (RAmin, RAmax)):
        sys.exit("ERROR: RA_min and RA_max must be between 0 and 360 degrees")
    if DECmax <= DECmin:
        sys.exit("ERROR: DEC_min must be lower than DEC_max")
    # check pointings related arguements
    if args.pointings_file is not None and args.grid is None:
        parser.error(
            "--pointings-file requires --grid")
    if args.grid is not None and args.pointings_file is None:
        parser.error(
            "--grid requires --pointings-file")

    # create footprint.txt file that lists the RA/DEC boundaries of this
    # survey (and others created with this script in the same folder)
    area = footprint_area(*args.bounds)
    print("registering survey in: %s" % args.footprint_file)
    if os.path.exists(args.footprint_file):
        surveys = read_footprint_file(args.footprint_file)
        if args.survey in surveys:  # update with current area
            print(
                "WARNING: survey '%s' already registered, updating values" %
                args.survey)
        surveys[args.survey] = (*args.bounds, area)
    else:
        surveys = {args.survey: (*args.bounds, area)}

    # rewrite the footprint file with possibly updated footprint
    with open(args.footprint_file, "w") as f:
        f.write(
            "# RA min/max               DEC min/max                " +
            "AREA             FIELD NAME\n")
        for survey, props in surveys.items():
            ra_min, ra_max, dec_min, dec_max, area = props
            f.write(
                "%11.7f %11.7f    %+11.7f %+11.7f    %13.7e    %s\n" % (
                    ra_min, ra_max, dec_min, dec_max, area, survey))

    # create the pointings file
    if args.pointings_file is not None:
        # get the grid shape
        pointings_ra, pointings_dec = args.grid
        n_pointings = pointings_ra * pointings_dec
        area = footprint_area(RAmin, RAmax, DECmin, DECmax) / n_pointings
        # split the footprint into equal RA columns
        if RAmax >= RAmin:
            RAs = np.linspace(RAmin, RAmax, pointings_ra + 1)
        else:
            RAs = np.linspace(RAmin, RAmax + 360.0, pointings_ra + 1)
            RAs[RAs >= 360.0] -= 360.0
        ra_mins = RAs[:-1]
        ra_maxs = RAs[1:]
        # split the RA columns in DEC rows such that all pointings have the
        # same area (up to rounding errors)
        DECs = [DECmin]
        print(
            "create %d x %d = %d pointings with %.7e sqdeg each" % (
                pointings_ra, pointings_dec, n_pointings, area))
        for i in range(pointings_dec):
            # calculate the next declination cut using the width in RA and
            # the targeted pointing area
            DECs.append(next_DEC(area, RAs[0], RAs[1], DECs[-1]))
        dec_mins = DECs[:-1]
        dec_maxs = DECs[1:]
        # generate a file that defines the pointing boundaries
        print("write pointing file: %s" % args.pointings_file)
        # combine the RA/DEC bounds
        bound_tuples = [
            (*ras, *decs) for decs, ras in product(
                zip(dec_mins, dec_maxs), zip(ra_mins, ra_maxs))]
        pointing_names = [
            pointing_name(args.survey, *bound_tuple)
            for bound_tuple in bound_tuples]
        name_len = max(len(name) for name in pointing_names)
        with open(args.pointings_file, "w") as f:
            for bounds, name in zip(bound_tuples, pointing_names):
                # line format: tile name, RAmin, RAmax, DECmin, DECmax
                f.write(
                    "%s    %11.7f    %11.7f    %+11.7f    %+11.7f\n" % (
                        name.ljust(name_len), *bounds))
