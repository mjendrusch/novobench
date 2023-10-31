# copyright (c) 2023 Michael Jendrusch, EMBL
# AlphaDesign: A de novo protein design framework based on AlphaFold
# Michael Jendrusch, Alessio Ling Jie Yang, Elisabetta Cacace, Jacob Bobonis,
# Carlos Geert Pieter Voogdt, Athanasios Typas, Jan O. Korbel, and S. Kashif Sadiq

import os
from typing import Optional, Iterable, List

def listdir_nohidden(path: str, extensions: Optional[Iterable[str]] = None) -> List[str]:
    r"""Lists the contents of a directory, skipping over hidden files.
    Args:
        path (str): path to directory to list files of.
        extensions (Optional[Iterable[str]]): list of admissible file-extensions.
            Default: no restriction on file extensions
    Returns:
        List of files and directories in `path`, skipping hidden files as well
        as files without the specified file extensions.
        The list is returned in alphabetical order.
    """
    return sorted([
        name for name in os.listdir(path)
        if ((name.split(".")[-1] in extensions) if extensions else True)
           and not name.startswith(".")])