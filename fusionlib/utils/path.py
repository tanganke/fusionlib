import os
from pathlib import Path
from typing import List, Union


def listdir_fullpath(dir: Union[str, Path]) -> Union[List[str], List[Path]]:
    """list directory `dir`, return fullpaths

    Args:
        dir (Union[str, Path]): directory name or Path object

    Returns:
        Union[List[str], List[Path]]: a list of fullpaths
    """
    assert os.path.isdir(dir), "Argument 'dir' must be a Directory"
    if isinstance(dir, str):
        names = os.listdir(dir)
        return [os.path.join(dir, name) for name in names]
    elif isinstance(dir, Path):
        names = dir.iterdir()
        return [dir / i for i in names]
    else:
        raise TypeError("Argument 'dir' must be a string or a Path object")
