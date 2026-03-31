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
    assert len(results) > 0, "no matching datadict found"
    if not newest:
        if acquire_time is None: 
            assert len(results) == 1, f"{len(results)} matching datadicts found"
            return max(results, key=itemgetter(0))
        else:
            for result in results:
                if acquire_time == result[0][11:17]:
                    return result
            raise AssertionError("there are no datadicts")