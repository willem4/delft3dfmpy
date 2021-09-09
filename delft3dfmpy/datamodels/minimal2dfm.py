import os
import pickle

import geopandas as gpd
import pandas as pd

from delft3dfmpy.datamodels.common import ExtendedDataFrame, ExtendedGeoDataFrame
from shapely.geometry import LineString, Point, Polygon

class Minimal2FM:
    """
    Minimal class to D-Flow FM datamodel
    """

    def __init__(self, extent_file=None):

        # Read geometry to clip data
        if extent_file is not None:
            self.clipgeo = gpd.read_file(extent_file).unary_union
        else:
            self.clipgeo = None
            
        # Create standard dataframe for network, crosssections, structures
        self.branches = ExtendedGeoDataFrame(geotype=LineString, required_columns=[
            'id',
            'geometry'
        ])
        
        self.crosssections = ExtendedGeoDataFrame(geotype=Point, required_columns=[
            "id",
            "width", 
            "depth", 
            "geometry"
        ])
        
    def to_pickle(self, filename, overwrite=False):
        # Check if path exists
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'File "{filename}" already exists.')

        # Dump object
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle)

    @classmethod
    def from_pickle(cls, filename):
        # Read object
        with open(filename, 'rb') as handle:
            loaded_cls = pickle.load(handle)

        return loaded_cls


