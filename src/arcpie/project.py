from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from arcpy.mp import ArcGISProject

if TYPE_CHECKING:
    from arcpy._mp import (
        Map,
        Layout,
        Layer,
        Table,
    )


class Project:
    def __init__(self, aprx_path: str|Path|Literal['CURRRENT']='CURRENT') -> None:
        self.path = str(aprx_path)
        self._aprx: ArcGISProject|None = None
    
    @property
    def aprx(self) -> ArcGISProject:
        if self._aprx is None:
            self._aprx = ArcGISProject(self.path)
        return self._aprx
    
    @property
    def maps(self) -> list[Map]:
        return self.aprx.listMaps()
    
    @property
    def layouts(self) -> list[Layout]:
        return self.aprx.listLayouts()
    
    @property
    def broken_sources(self) -> list[Layer|Table]:
        return self.aprx.listBrokenDataSources() #type: ignore
    