from __future__ import annotations

import logging
import typing

from lgdo.lh5 import LH5Store
from lgdo.types import LGDO

log = logging.getLogger(__name__)


class ELMIterator:
    """A class to iterate over the rows of an event lookup map"""

    def __init__(
        self,
        elm_file: str,
        stp_file: str,
        lh5_group: str,
        start_row: int,
        n_rows: int | None,
        *,
        stp_field: str = "stp",
        read_vertices: bool = False,
        buffer: int = 10000,
    ):
        """Constructor for the ELMIterator.

        Parameters
        ----------
        elm_file
            the file containing the event lookup map.
        stp_file
            the file containing the steps to read.
        lh5_subgroup
            the name of the lh5 subgroup to read.
        start_row
            the first row to read.
        n_rows
            the number of rows to read, if `None` read them all.
        read_vertices
            whether to read also the vertices table.
        buffer
            the number of rows to read at once.

        """

        # initialise
        self.elm_file = elm_file
        self.stp_file = stp_file
        self.lh5_group = lh5_group
        self.start_row = start_row
        self.start_row_tmp = start_row
        self.n_rows = n_rows
        self.buffer = buffer
        self.current_i_entry = 0
        self.read_vertices = read_vertices
        self.stp_field = stp_field

        # would be good to replace with an iterator
        self.sto = LH5Store()
        self.n_rows_read = 0

    def __iter__(self) -> typing.Iterator:
        self.current_i_entry = 0
        self.n_rows_read = 0
        self.start_row_tmp = self.start_row
        return self

    def __next__(self) -> tuple[LGDO, LGDO | None, int, int]:
        # get the number of rows to read
        rows_left = self.n_rows - self.n_rows_read
        n_rows = self.buffer if (self.buffer > rows_left) else rows_left

        # read the elm rows
        elm_rows, n_rows_read = self.sto.read(
            f"elm/{self.lh5_group}", self.elm_file, start_row=self.start_row_tmp, n_rows=n_rows
        )

        self.n_rows_read += n_rows_read
        self.start_row_tmp += n_rows_read

        if n_rows_read == 0:
            raise StopIteration

        # view our elm as an awkward array
        elm_ak = elm_rows.view_as("ak")

        # remove empty rows
        elm_ak = elm_ak[elm_ak.n_rows > 0]

        if len(elm_ak) > 0:
            # extract range of stp rows to read
            start = elm_ak.start_row[0]
            n = sum(elm_ak.n_rows)

            stp_rows, n_steps = self.sto.read(
                f"{self.stp_field}/{self.lh5_group}", self.stp_file, start_row=start, n_rows=n
            )

            self.current_i_entry += 1

            if self.read_vertices:
                vert_rows, _ = self.sto.read(
                    f"{self.stp_field}/vertices",
                    self.stp_file,
                    start_row=self.start_row,
                    n_rows=n_rows,
                )
            else:
                vert_rows = None
            # vertex table should have same structure as elm

            return (stp_rows, vert_rows, self.current_i_entry, n_steps)
        return (None, None, self.current_i_entry, 0)
