import json
import os
import pathlib as pl
import re
from datetime import datetime
from typing import List, Union

import numpy as np
from defaults import get_available_workspaces


class SimulationData:
    def __init__(self, file_name: os.PathLike):
        # Set up file reading
        if isinstance(file_name, str):
            file_name = pl.Path(file_name)
        if not file_name.is_file():
            raise FileNotFoundError(f"file_name `{file_name}` not found")
        self.file_name = file_name
        self.f = open(file_name, "r", encoding="ascii", errors="replace")

    def is_normal_termination(self) -> bool:
        """
        Determine if the simultion terminated normally

        Returns
        -------
        success: bool
            Boolean indicating if the simulation terminated normally

        """
        # rewind the file
        self.f.seek(0)

        seekpoint = self._seek_to_string("Normal termination of simulation.")
        self.f.seek(seekpoint)
        line = self.f.readline()
        if line == "":
            success = False
        else:
            success = True
        return success

    def get_model_runtime(self, units: str = "seconds") -> float:
        """
        Get the elapsed runtime of the model from the list file.

        Parameters
        ----------
        units : str
            Units in which to return the runtime. Acceptable values are
            'seconds', 'minutes', 'hours' (default is 'seconds')

        Returns
        -------
        out : float
            Floating point value with the runtime in requested units. Returns
            NaN if runtime not found in list file

        """
        # rewind the file
        self.f.seek(0)

        units = units.lower()
        if (
            not units == "seconds"
            and not units == "minutes"
            and not units == "hours"
        ):
            raise AssertionError(
                '"units" input variable must be "minutes", "hours", '
                f'or "seconds": {units} was specified'
            )

        seekpoint = self._seek_to_string("Elapsed run time:")
        self.f.seek(seekpoint)
        line = self.f.readline()
        if line == "":
            return np.nan

        # yank out the floating point values from the Elapsed run time string
        times = list(map(float, re.findall(r"[+-]?[0-9.]+", line)))
        # pad an array with zeros and times with
        # [days, hours, minutes, seconds]
        times = np.array([0 for _ in range(4 - len(times))] + times)
        # convert all to seconds
        time2sec = np.array([24 * 60 * 60, 60 * 60, 60, 1])
        times_sec = np.sum(times * time2sec)
        # return in the requested units
        if units == "seconds":
            return times_sec
        elif units == "minutes":
            return times_sec / 60.0
        elif units == "hours":
            return times_sec / 60.0 / 60.0

    def get_formulate_time(self) -> float:
        """
        Get the formulate time for the solution from the list file.

        Returns
        -------
        out : float
            Floating point value with the formulate time,

        """
        # rewind the file
        self.f.seek(0)

        try:
            seekpoint = self._seek_to_string("Total formulate time:")
        except:
            print(
                "'Total formulate time' not included in list file. "
                + "Returning NaN"
            )
            return np.nan

        self.f.seek(seekpoint)
        return float(self.f.readline().split()[3])

    def get_solution_time(self) -> float:
        """
        Get the solution time for the solution from the list file.

        Returns
        -------
        out : float
            Floating point value with the solution time,

        """
        # rewind the file
        self.f.seek(0)

        try:
            seekpoint = self._seek_to_string("Total solution time:")
        except:
            print(
                "'Total solution time' not included in list file. "
                + "Returning NaN"
            )
            return np.nan

        self.f.seek(seekpoint)
        return float(self.f.readline().split()[3])

    def get_outer_iterations(self) -> int:
        """
        Get the total outer iterations from the list file.

        Parameters
        ----------

        Returns
        -------
        outer_iterations : float
            Sum of all TOTAL ITERATIONS found in the list file

        """
        # initialize total_iterations
        outer_iterations = 0

        # rewind the file
        self.f.seek(0)

        while True:
            seekpoint = self._seek_to_string("CALLS TO NUMERICAL SOLUTION IN")
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            outer_iterations += int(line.split()[0])

        return outer_iterations

    def get_total_iterations(self) -> int:
        """
        Get the total number of iterations from the list file.

        Parameters
        ----------

        Returns
        -------
        total_iterations : float
            Sum of all TOTAL ITERATIONS found in the list file

        """
        # initialize total_iterations
        total_iterations = 0

        # rewind the file
        self.f.seek(0)

        while True:
            seekpoint = self._seek_to_string("TOTAL ITERATIONS")
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            total_iterations += int(line.split()[0])

        return total_iterations

    def get_memory_usage(self, virtual=False) -> float:
        """
        Get the simulation memory usage from the simulation list file.

        Parameters
        ----------

        Returns
        -------
        memory_usage : float
            Total memory usage for a simulation (in Gigabytes)

        """
        # initialize total_iterations
        memory_usage = 0.0

        # rewind the file
        self.f.seek(0)

        tags = (
            "MEMORY MANAGER TOTAL STORAGE BY DATA TYPE",
            "Total",
            "Virtual",
        )

        while True:
            seekpoint = self._seek_to_string(tags[0])
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            units = line.split()[-1]
            if units == "GIGABYTES":
                conversion = 1.0
            elif units == "MEGABYTES":
                conversion = 1e-3
            elif units == "KILOBYTES":
                conversion = 1e-6
            elif units == "BYTES":
                conversion = 1e-9
            else:
                raise ValueError(f"Unknown memory unit '{units}'")

            if virtual:
                tag = tags[2]
            else:
                tag = tags[1]
            seekpoint = self._seek_to_string(tag)
            self.f.seek(seekpoint)
            line = self.f.readline()
            if line == "":
                break
            memory_usage = float(line.split()[-1]) * conversion

        return memory_usage

    def get_non_virtual_memory_usage(self):
        """

        Returns
        -------
        non_virtual: float
            Non-virtual memory usage, which is the difference between the
            total and virtual memory usage

        """
        return self.get_memory_usage() - self.get_memory_usage(virtual=True)

    def _seek_to_string(self, s):
        """
        Parameters
        ----------
        s : str
            Seek through the file to the next occurrence of s.  Return the
            seek location when found.

        Returns
        -------
        seekpoint : int
            Next location of the string

        """
        while True:
            seekpoint = self.f.tell()
            line = self.f.readline()
            if line == "":
                break
            if s in line:
                break
        return seekpoint


def get_simulation_processors(
    metis: bool = False, simulation_type: str = "basin_structured"
) -> List[int]:
    """
    Get a processor combinations from the list of available workspaces.

    Parameters
    ----------
    metis : bool
        Boolean that indicates if searching for metis simulations. metis can
        not be True if voronoi is True. (Default is False)
    simulation_type: str
        name of simulation. Must be "basin_structured", "basin_unstructured",
        or "box_structured". (Default is "basin_structured")

    Returns
    -------
    processors: list of ints
        Available processor simulations

    """
    workspaces = get_available_workspaces(
        metis=metis,
        simulation_type=simulation_type,
    )
    processors = [int(workspace.name[-4:-1]) for workspace in workspaces[1:]]
    return [1] + sorted(processors)


def get_simulation_listfiles(path: os.PathLike) -> list:
    """
    Get all simulation list files in a path

    Parameters
    ----------
    path: PathLike
        path to simulation data

    Returns
    -------
    list_files: list
        list containing all listing files matching mfsim*.lst pattern

    """
    if isinstance(path, str):
        path = pl.Path(path)

    list_files = []
    for file in path.glob("mfsim*.lst"):
        list_files.append(path / file.name)
    return list_files


def get_available_json_files(
    path: os.PathLike = "performance",
) -> List[os.PathLike]:
    """
    Get a list of available json files
    Parameters
    ----------
    path: PathLike
        path to simulation data

    Returns
    -------
    json_paths: os.PathLike
        paths for available json files in path directory

    """
    if isinstance(path, str):
        path = pl.Path(path)
    json_paths = []
    for file in path.glob("*.json"):
        json_paths.append(path / file.name)
    return json_paths


def list_available_json_files(path: os.PathLike = "performance") -> None:
    """
    Print a list of available json files in path directory
    Parameters
    ----------
    path: PathLike
        path to simulation data

    Returns
    -------
    None

    """
    if isinstance(path, str):
        path = pl.Path(path)
    file_paths = get_available_json_files(path)
    for file_path in file_paths:
        print(f"{file_path}")


def save_performance_json(
    performance_dict: dict,
    filename: str = None,
    workspace: Union[os.PathLike, str] = "performance",
) -> os.PathLike:
    """

    Parameters
    ----------
    performance_dict
    filename
    workspace

    Returns
    -------
    file_path: os.PathLike
        Performance

    """
    if not isinstance(workspace, pl.Path):
        workspace = pl.Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if filename is None:
        if "simulation_type" in performance_dict.keys():
            filename = performance_dict["simulation_type"]
            if "unstructured" in performance_dict["simulation_type"]:
                performance_dict["metis"] = False
        else:
            filename = performance_dict["grid_type"]
        if performance_dict["metis"]:
            filename = f"{filename}_metis"
        filename = (
            f"{filename}_"
            + f"{performance_dict['total_cells']:010d}_"
            + f"{performance_dict['active_cells']:010d}.json"
        )
    file_path = workspace / filename

    # update performance dictionary with current date and time
    performance_dict["created"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    print(f"Writing performance data to '{file_path}'.")
    with open(file_path, "w") as f:
        json.dump(performance_dict, f)

    return file_path


def load_performance_json(
    filename: str,
    workspace: Union[str, os.PathLike] = "performance",
) -> dict:
    """

    Parameters
    ----------
    filename: str
        Performance data json file name. The .json extension is added to
        filename if it is not included.
    workspace: str or os.PathLike
        path to directory containing the performance json file

    Returns
    -------
    performance: dict
        Performance data dictionary

    """
    if not isinstance(workspace, pl.Path):
        workspace = pl.Path(workspace)
    if not filename.lower().endswith(".json"):
        filename += ".json"
    file_path = workspace / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"'{file_path}' does not exist")

    print(f"Loading performance data from '{file_path}'.")
    with open(file_path) as f:
        performance = json.load(f)
    return performance
