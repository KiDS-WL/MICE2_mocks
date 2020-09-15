#
# This module implements a bit mask manager used to store information for
# sample selections. Each bit represents an independent sub-set of the
# selection function.
#

from collections import OrderedDict

import numpy as np


class BitMaskManager(object):

    _bit_type_map = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _base_description = "{:} sample selection bit mask, select (sub-)samples "
    _base_description += "by (mask & bit), where the bit represents: "

    def __init__(self, sample, nbits=8):
        if nbits not in self._bit_type_map.keys():
            message = "number of bits must be either of: {:}".format(
                sorted(self._bit_type_map.keys))
            raise ValueError(message)
        self._sample = sample
        self._bit_desc = OrderedDict()
        # indicate which bits are occupied
        self._bit_reserv = OrderedDict(
            (2**i, False) for i in reversed(range(nbits)))
        # reserve the lowest bit for easy selection of the full sample
        self._bit_reserv[1] = True
        self._bit_desc[1] = "full sample"

    def __str__(self):
        bitstring = "".join(
            "1" if reserved else "0" for reserved in self._bit_reserv.values())
        return "BitMask({:})".format(bitstring)

    @property
    def dtype(self):
        return self._bit_type_map[len(self._bit_reserv)]

    @property
    def available(self):
        return tuple(
            bit for bit, reserved in self._bit_reserv.items() if not reserved)

    @property
    def reserved(self):
        return tuple(
            bit for bit, reserved in self._bit_reserv.items() if reserved)

    @property
    def description(self):
        bit_descs = []
        for bit in self.reserved:
            bit_descs.append("({:d}) {:}".format(bit, self._bit_desc[bit]))
        # build the documentation string
        string = "{:} sample selection bit mask, select ".format(self._sample)
        string += "(sub-)samples by (mask & bit), where the bits represent: "
        string += ", ".join(reversed(bit_descs))
        return string

    def reserve(self, description):
        if all(self._bit_reserv.values()):
            raise ValueError("all bits are reserved")
        # get the lowest, unreserved bit
        bit = self.available[-1]
        self.reserve_bit(bit, description)
        return bit

    def reserve_bit(self, bit, description):
        if all(self._bit_reserv.values()):
            raise ValueError("all bits are reserved")
        try:
            if self._bit_reserv[bit]:
                raise ValueError("bit is already reserved: {:d}".format(bit))
            else:
                self._bit_reserv[bit] = True
                self._bit_desc[bit] = description
        except KeyError:
            raise KeyError("invalid bit value: {:}".format(bit))

    @staticmethod
    def check_bit(bitmask, bit):
        # ensure same data type
        bit = bitmask.dtype.type(bit)
        # check if bit == 1
        is_set = (bitmask & bit).astype(bool)
        return is_set

    @staticmethod
    def check_master(bitmask):
        return BitMaskManager.check_bit(bitmask, 1)

    @staticmethod
    def check_bits_all(bitmask, bit_sum):
        # ensure same data type
        bit_sum = bitmask.dtype.type(bit_sum)
        # check if bit == 1
        all_set = (bitmask & bit_sum) == bit_sum
        return all_set

    @staticmethod
    def check_bits_any(bitmask, bit_sum):
        # ensure same data type
        bit_sum = bitmask.dtype.type(bit_sum)
        # check if bit == 1
        any_set = (bitmask & bit_sum) > 0
        return any_set

    @staticmethod
    def set_bit(bitmask, bit, condition=None, copy=False):
        # ensure correct data type
        if condition is None:  # set all bits
            bits = bitmask.dtype.type(bit)
        else:  # set to bit where condition is True, otherwise zero
            bits = np.where(
                condition, bitmask.dtype.type(bit), bitmask.dtype.type(0))
        # set bit
        updated = bitmask | bits
        if copy:
            return updated
        else:
            bitmask[:] = updated
            return bitmask

    @staticmethod
    def clear_bit(bitmask, bit, condition=None, copy=False):
        # ensure correct data type
        if condition is None:  # set all bits
            bits = bitmask.dtype.type(bit)
        else:  # set to bit where condition is True, otherwise zero
            bits = np.where(
                condition, bitmask.dtype.type(bit), bitmask.dtype.type(0))
        # clear bit
        updated = bitmask & ~bits
        if copy:
            return updated
        else:
            bitmask[:] = updated
            return bitmask

    @staticmethod
    def toggle_bit(bitmask, bit, condition=None, copy=False):
        # ensure correct data type
        if condition is None:  # set all bits
            bits = bitmask.dtype.type(bit)
        else:  # set to bit where condition is True, otherwise zero
            bits = np.where(
                condition, bitmask.dtype.type(bit), bitmask.dtype.type(0))
        # flip bit
        updated = bitmask ^ bit
        if copy:
            return updated
        else:
            bitmask[:] = updated
            return bitmask

    @staticmethod
    def place_bit(bitmask, bit, values, copy=False):
        # ensure same data type
        bit = bitmask.dtype.type(bit)
        # first reduce values to 0 or 1, then to 0 or bit
        values = np.asarray(values, dtype="bool").astype(bitmask.dtype)
        values *= bit
        # set values
        updated = (bitmask & ~bit) | values
        if copy:
            return updated
        else:
            bitmask[:] = updated
            return bitmask

    @staticmethod
    def place_master(bitmask, values, copy=False):
        return BitMaskManager.place_bit(bitmask, 1, values, copy)

    @staticmethod
    def update_master(bitmask, bit_sum, bit_join="AND", copy=False):
        assert(bit_join in ("AND", "OR"))
        # join the selected subset of bits
        if bit_join == "AND":
            is_selected = BitMaskManager.check_bits_all(bitmask, bit_sum)
        else:
            is_selected = BitMaskManager.check_bits_any(bitmask, bit_sum)
        # check if the master bit is set and update it accordingly
        is_selected &= BitMaskManager.check_master(bitmask)
        updated = BitMaskManager.place_master(bitmask, is_selected, copy=copy)
        if copy:
            return updated
        else:
            bitmask[:] = updated
            return bitmask

    @staticmethod
    def print_binary(number, width=8):
        try:
            print(np.binary_repr(number, width))
        except TypeError:
            for num in number:
                print(np.binary_repr(num, width))
