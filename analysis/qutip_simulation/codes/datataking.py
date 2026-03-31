from plottr.data.datadict_storage import *

def search_datadict_miyamura(
    basedir: Union[str, Path],
    since: str,
    until: Optional[str] = None,
    acquire_time: str = None,
    name: Optional[str] = None,
    groupname: str = 'data',
    filename: str = 'data',
    structure_only: bool = False,
    only_complete: bool = True,
    skip_trash: bool = True,
    newest: bool = False,
) -> Tuple[str, DataDict]:
    """Find the datadict which matches a set of conditions.
    `AssertionError` is raised if there are zero or multiple matching datadicts.

    :param basedir: The root directory in which data is stored.
    :param since: Date (and time) in the format `YYYY-mm-dd` (or `YYYY-mm-ddTHHMMSS`).
    :param until: Date (and time) in the format `YYYY-mm-dd` (or `YYYY-mm-ddTHHMMSS`).
        If not given, default to `until = since`.
    :param acquire_time: Time the data was acquired (6 digits after 'T' in the filename)
    :param name: Name of the dataset (if not given, match all datasets).
    :param groupname: Name of hdf5 group.
    :param filename: Name of the ddh5 file without the extension.
    :param structure_only: If `True`, don't load the data values.
    :param only_complete: If `True`, only return datadicts tagged as complete.
    :param skip_trash: If `True`, skip datadicts tagged as trash.
    :param newest: If `True`, return the newest matching datadict
    :return: (foldername, datadict).
    """
    results = list(search_datadicts(
        basedir,
        since,
        until=until,
        name=name,
        groupname=groupname,
        filename=filename,
        structure_only=structure_only,
        only_complete=only_complete,
        skip_trash=skip_trash,
    ))

    
    assert len(results) > 0, "no matching datadict found1"
    if not newest:
        if acquire_time is None: 
            assert len(results) == 1, f"{len(results)} matching datadicts found"
            return max(results, key=itemgetter(0))
        else:
            for result in results:
                if acquire_time == result[0][11:17]:
                    return result
            raise AssertionError(f"there are no datadicts:{since}-{acquire_time}")
