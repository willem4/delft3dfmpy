# Basis
import os
import sys
import shutil
import numpy as np
sys.path.append(r'../')
# Importing relevant classes from delft3dfmpy
import delft3dfmpy

from delft3dfmpy import Minimal2FM, DFlowFMModel, Rectangular, DFlowFMWriter
from delft3dfmpy.datamodels.common import ExtendedGeoDataFrame
from delft3dfmpy.core.logging import initialize_logger

import configparser, json

# For reading SOBEK results as boundary conditions
# hkvsobekpy requires the modules fire and tqdm, install these (conda install fire tqdm)
# import hkvsobekpy

# shapefiles IO
import geopandas as gpd

# Import csv
import pandas as pd

# Geometries
from shapely.geometry import Polygon, LineString

# Helper function 

def translate_data(config, topic):
    """Translate data to correct format with required columns and 
    correct projection.
    """    
    # Generate empty to inquire to required data columns
    fm_dummy = Minimal2FM()

    # Read source data 
    umap = os.path.join(data_path, config.get(topic, 'file'))
    umap_df = gpd.read_file(umap)
    make_path(os.path.join(data_path, 'temp'))
    fn = os.path.join(data_path, 'temp', topic+'.shp')
    gdf = umap_df.to_crs(config.get('parameter', 'projectedcrs'))

    # Perform mapping
    attributes = getattr(fm_dummy, topic)
    if hasattr(attributes, 'columns'):
        for required_column in getattr(attributes,'columns'):
            gdf.rename(columns = {config.get(topic, required_column):required_column}, inplace = True)
            
    # Save data and return new file location
    gdf.to_file(fn)
    return fn 



# path to dflowfm for refining the mesh 
dflowfm_path = r'd:/current/dflowfm'

# path to write the models
output_path = r'../modellen'

# path to DIMR for a correct run.bat file
dimr_path = r'd:/current/dimr/bin/dimr/scripts/run_dimr.bat'

def make_path(pathdir): 
    if not os.path.exists(pathdir):
        os.mkdir(pathdir)

assert ('test-1672.py' in os.listdir()), 'Expecting current workdirectory to contain test-1672.py'


root = os.path.abspath('../data/1672')
fn_ini = os.path.join(root, 'osm_settings.ini')

logger = initialize_logger('1672osm2fm.log', log_level=10)

# Read ini file
logger.info(f'Using delft3dfmpy {delft3dfmpy.__version__}')
logger.info(f'Read config from {fn_ini}')
config = configparser.ConfigParser(inline_comment_prefixes=[";", "#"])
config.read(fn_ini)

# path to the package containing the dummy-data
data_path = config.get('input', 'DataPath')

# Get parameters
parameters = config._sections['parameter']

fn_study_area = translate_data(config, 'clipgeo')
fn_branches = translate_data(config, 'branches')
fn_crosssections = translate_data(config, 'crosssections')

# fn_crosssections = os.path.join(data_path, 'umap', 'profiles.geojson')
# fn_profiles = os.path.join(data_path, 'gml', 'NormGeparametriseerdProfiel.gml')
# fn_bridges = os.path.join(data_path, 'gml', 'brug.gml')
# fn_culverts = os.path.join(data_path, 'gml', 'duikersifonhevel.gml')
# fn_weirs = os.path.join(data_path, 'gml', 'stuw.gml')
# fn_orifices = os.path.join(data_path, 'gml', 'onderspuier.gml')
# fn_valves = os.path.join(data_path, 'gml', 'afsluitmiddel.gml')
# fn_laterals = os.path.join(data_path, 'sobekdata', 'Sbk_S3BR_n.shp')
# fn_pump1 = os.path.join(data_path, 'gml', 'gemaal.gml')
# fn_pump2 = os.path.join(data_path, 'gml', 'pomp.gml')
# fn_control = os.path.join(data_path, 'gml', 'sturing.gml')


fm_data = Minimal2FM(extent_file=fn_study_area)


# # Branches
fm_data.branches.read_shp(fn_branches, id_col='id', clip=fm_data.clipgeo)
fm_data.crosssections.read_shp(fn_branches, id_col='id', clip=fm_data.clipgeo)
#osm.branches['ruwheidstypecode'] = 4

# read cross sections from GML
# hydamo.crosssections.read_gml(fn_crosssections, 
#                               column_mapping={'ruwheidswaardelaag':'ruwheidswaarde'} ,
#                               index_col='profielcode' ,
#                               groupby_column='profielcode' , 
#                               order_column='codevolgnummer')

# hydamo.crosssections.snap_to_branch(hydamo.branches, snap_method='intersecting')
# hydamo.crosssections.dropna(axis=0, inplace=True, subset=['branch_offset'])
# hydamo.crosssections.drop('code', axis=1, inplace=True)
# hydamo.crosssections.rename(columns={'profielcode': 'code'}, inplace=True)

# hydamo.parametrised_profiles.read_gml(fn_profiles, column_mapping={'ruwheidswaardelaag': 'ruwheidswaarde'})
# hydamo.parametrised_profiles.snap_to_branch(hydamo.branches, snap_method='intersecting')
# hydamo.parametrised_profiles.dropna(axis=0, inplace=True, subset=['branch_offset'])

# # Bridges
# hydamo.bridges.read_gml(fn_bridges)
# hydamo.bridges.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=5)
# hydamo.bridges.dropna(axis=0, inplace=True, subset=['branch_offset'])

# # Culverts
# hydamo.culverts.read_gml(
#    fn_culverts,
#    index_col='code',
#    column_mapping={'vormkoker': 'vormcode'},
#    clip=hydamo.clipgeo
# )
# hydamo.culverts.snap_to_branch(hydamo.branches, snap_method='ends', maxdist=5)
# hydamo.culverts.dropna(axis=0, inplace=True, subset=['branch_offset'])
# duikers_rekentijd = ['RS372-KDU3','RS375-KDU2','RS373-KDU7','RS373-KDU20','RS373-KDU22','RS373-KDU19']
# duikers_gemalen = ['OWL32921-KDU3','RS375-KDU6']             
# hydamo.culverts.drop(duikers_rekentijd, axis=0, inplace=True)
# hydamo.culverts.drop(duikers_gemalen, axis=0, inplace=True)

# # Weirs (including universal weirs)
# hydamo.weirs.read_gml(fn_weirs)
# hydamo.weirs.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=10)
# hydamo.weirs.dropna(axis=0, inplace=True, subset=['branch_offset'])

# # Orifices
# hydamo.orifices.read_gml(fn_orifices)
# hydamo.orifices.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=2)
# hydamo.orifices.dropna(axis=0, inplace=True, subset=['branch_offset'])

# # Closing devices / terugslagkleppen e.d.
# hydamo.afsluitmiddel.read_gml(fn_valves, index_col='code')

# # Laterals (imported from shapefile)
# hydamo.laterals.read_shp(fn_laterals,
#                          column_mapping={'ID        ': 'code', 
#                                          'NAME      ': 'name',
#                                          'X         ':'X',
#                                          'Y         ':'Y'})
# hydamo.laterals.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=  5)
# hydamo.laterals.dropna(axis=0, inplace=True, subset=['branch_offset'])

# # Pumps
# hydamo.gemalen.read_gml(fn_pump1, index_col='code', clip=hydamo.clipgeo)
# hydamo.pumps.read_gml(fn_pump2, index_col='code', clip=hydamo.clipgeo)
# hydamo.pumps.snap_to_branch(hydamo.branches, snap_method='overal', maxdist=5)
# hydamo.pumps['maximalecapaciteit'] *= 60
# hydamo.sturing.read_gml(fn_control, index_col='code')


# Plot the model with branches, cross sections and structures. Note that compound structures are not plotted here as they do not have a geometry on their own; they are composed from their sub-structures that do have a geometry and are plotted here.

# Plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

fn_background = os.path.join(data_path, 'gis', 'ohw-06-656x1024.jpg')
plt.rcParams['axes.edgecolor'] = 'w'

fig, ax = plt.subplots(figsize=(10, 10))

ax.fill(*fm_data.clipgeo.exterior.xy, color='w', alpha=0.5)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_xlim(95032.2496803394606104,145054.9318180521368049)
ax.set_ylim(423138.1493090987205505,446548.8388533919351175)

achtergrond = plt.imread(fn_background)
ax.imshow(achtergrond, extent=(95032.2496803394606104,145054.9318180521368049,423138.1493090987205505,501148.8388533919351175), interpolation='lanczos')

fm_data.branches.plot(ax=ax, label='Channel')
# hydamo.crosssections.plot(ax=ax, color='C3', label='Cross section')
# hydamo.culverts.centroid.plot(ax=ax, color='darkgreen', label='Culvert', markersize=20, zorder=10)
# hydamo.weirs.centroid.plot(ax=ax, color='C1', label='Weir', markersize=25, zorder=10)
# hydamo.bridges.plot(ax=ax,color='red',label='Bridge',markersize=20,zorder=10)
# hydamo.orifices.plot(ax=ax,color='black',label='Orifice',markersize=20,zorder=10)
# hydamo.pumps.plot(
#     ax=ax, color='C4', label='Pump', marker='s', markersize=125, zorder=10, facecolor='none', linewidth=2.5)

ax.legend()

fig.tight_layout()
plt.savefig('input.png')


# ### Generate the D-HYDRO FM schematisation

# #### Create the 1D network

# Convert the geometries to D-HYDRO schematisation:
# 
# Start with importing the structures (from HyDAMO in this case), since the position of the structure can be used in defining the position of the 1d nodes. 
# 
# Structures can also be added without the HyDAMO imports. One weir is added manually, but this can be done for all implemented structures.
# 
# Note that for importing most structures multiple gml-files are needed. For more info on how to add structures (directly or from HyDAMO), see: https://hkvconfluence.atlassian.net/wiki/spaces/DHYD/overview.
# 
#  - for weirs, a corresponding profile is looked up in the crossections. If one is found (either a YZ or a parametrised profile) the weir is implemented as a universal weir. If it is not found, a regular (rectangular) weir will be used. The cross-section should contain a 'codegeralateerdobject' containing the ID of the universal weir.
#  - culverts can also use an 'afsluitmiddel'; if one is coupled for a specific culvert and its type is 5 (terugslagklep) the flow direction is set 'positive' instead of 'both'. If the type is 4 (schuif), a valve will be implemented.
#  - bridges need an associated crosssection (through the field 'codegerelateerdobject' in the cross-section); this can be either 'YZ' or 'parametrised'. The profiles are then processed so a suitable cross-section for a bridge is created;
#  - pumps are composed from 'gemalen', 'pompen' and 'sturing'.
# 
# In most cases, these 'extra' arguments are optional, i.e. they are not required and can be left out. Some are required:
# - pumps really need all 3 objects (gemalen, pompen en sturing);
# - bridges really need a profile (either 'crosssections' or 'parametrised_profiles' needs to contain a field 'codegerelateerdobject' that points to each bridge).
# 
# For more info on the structure definitions one is referred to the D-Flow FM user manual: https://content.oss.deltares.nl/delft3d/manuals/D-Flow_FM_User_Manual.pdf.
# 
# Note that orifices do not yet have an appropriate/definitive definition in HYDAMO. To be able to use it, we now use a separate GML-definition ('onderspuier") but possibly this will be integrated in the definition for weirs. To be continued.
# 


dfmmodel = DFlowFMModel()

# Create a 1D schematisation
dfmmodel.network.set_branches(fm_data.branches)
dfmmodel.network.generate_1dnetwork(one_d_mesh_distance=float(parameters['grid1dresolution']), seperate_structures=True)


# If there are still missing cross sections left, add a default one. To do so add a cross section definition, and assign it with a vertical offset (shift).

# # Set a default cross section
default = dfmmodel.crosssections.add_rectangle_definition(
    height=float(parameters['defaultdepth']), width=float(parameters['defaultwidth']), closed=False, 
    roughnesstype=parameters['roughnesstype'], roughnessvalue=float(parameters['roughness1d']))
dfmmodel.crosssections.set_default_definition(definition=default, shift=-float(parameters['defaultdepth']))

# #### Add a 2D mesh

# To add a mesh, currently 2 options exist:

# 1) the converter can generate a relatively simple, rectangular mesh, with a rotation or refinement. Note that rotation _and_ refinement is currently not possible. In the section below we generate a refined 2D mesh with the following steps:
# 
# - Generate grid within a polygon. The polygon is the extent given to the HyDAMO model.
# - Refine along the main branch
# - Determine altitude from a DEM.
# 
# The 'refine'-method requires the dflowfm.exe executable. If this is not added to the system path, it can be provided in an optional argument to refine (dflowfm_path).

# Create mesh object
mesh = Rectangular()
cellsize = float(parameters['grid2dresolution'])

# Generate mesh within model bounds
mesh.generate_within_polygon(fm_data.clipgeo, cellsize=cellsize, rotation=0)

# # Refine the model (2 steps) along the main branch. To do so we generate a buffer around the main branch.
# buffered_branch = hydamo.branches.loc[['riv_RS1_1810', 'riv_RS1_264'], 'geometry'].unary_union.buffer(10)
# mesh.refine(polygon=[buffered_branch], level=[2], cellsize=cellsize, dflowfm_path=dflowfm_path)

# Determine the altitude from a digital elevation model
# rasterpath = '../gis/AHNdommel_clipped.tif'
# mesh.altitude_from_raster(rasterpath)

# The full DEM is not added to this notebook. Instead a constant bed level is used
mesh.altitude_constant(0.5, where='node')

# Add to schematisation
dfmmodel.network.add_mesh2d(mesh)


# 2) a more complex mesh can be created in other software (such as SMS) and then imported in the converter: (uncomment to activate)

# from dhydamo.core.mesh2d import Mesh2D
# mesh = Mesh2D()
# import the geometry
#mesh.geom_from_netcdf(r'T:\2Hugo\Grid_Roer_net.nc')
# fill every cell with an elevation value
#mesh.altitude_from_raster(rasterpath)
# and add to the model
#dfmmodel.network.add_mesh2d(mesh)


# #### Add the 1D-2D links

# For linking the 1D and 2D model, three options are available:
# 1. Generating links from each 1d node to the nearest 2d node.
# 2. Generating links from each 2d node to the nearest 1d node (intersecting==True)
# 3. Generating links from each 2d node to the nearest 1d node, while not allowing the links to intersect other cells (intersecting==True).
# 
# Intersecting indicates whether or not the 2D cells cross the 1D network (lateral versus embedded links).
# So, option 3 is relevant when there is no 2d mesh on top of the 1d mesh: the lateral links.
# 
# Note that for each option a maximum link length can be chosen, to prevent creating long (and perhaps unrealistic) links.

del dfmmodel.network.links1d2d.faces2d[:]
del dfmmodel.network.links1d2d.nodes1d[:]
dfmmodel.network.links1d2d.generate_1d_to_2d(max_distance=float(parameters['grid1d2dmaxdistance']))


fig, ax = plt.subplots(figsize=(13, 10))
ax.set_aspect(1.0)

segments = dfmmodel.network.mesh2d.get_segments()
ax.add_collection(LineCollection(segments, color='0.3', linewidths=0.5, label='2D-mesh'))
segments1d = dfmmodel.network.mesh1d.get_segments()
ax.add_collection(LineCollection(segments1d, color='r', linewidths=0.5, label='1D-mesh'))

links = dfmmodel.network.links1d2d.get_1d2dlinks()
ax.add_collection(LineCollection(links, color='k', linewidths=0.5))
ax.plot(links[:, :, 0].ravel(), links[:, :, 1].ravel(), color='k', marker='.', ls='', label='1D2D-links')

# for i, p in enumerate([buffered_branch]):
#     ax.plot(*p.exterior.xy, color='C3', lw=1.5, zorder=10, alpha=0.8, label='Refinement buffer' if i==0 else None)

ax.set_xlim(95032.2496803394606104,145054.9318180521368049)
ax.set_ylim(423138.1493090987205505,446548.8388533919351175)

achtergrond = plt.imread(fn_background)
ax.imshow(achtergrond, extent=(95032.2496803394606104,145054.9318180521368049,423138.1493090987205505,501148.8388533919351175), interpolation='lanczos')

ax.legend()
plt.savefig('mesh.png')


# ### Boundary conditions for FM
# 
# Add boundary conditions to external forcings from a SOBEK time series.

# fn_bcs = os.path.join(data_path, 'sobekdata', 'boundaryconditions.csv')
# bcs = pd.read_csv(fn_bcs, sep=';', index_col=0)
# bcs.index = pd.to_datetime(bcs.index)


# dfmmodel.external_forcings.add_boundary_condition(
#     name='BC_flow_in',
#     pt=(140712.056047, 391893.277878),
#     bctype='discharge',
#     series=bcs['Discharge']
# )

# dfmmodel.external_forcings.add_boundary_condition(
#     name='BC_wlev_down',
#     pt=(141133.788766, 395441.748424),
#     bctype='waterlevel',
#     series=bcs['Waterlevel']
# )


# fig, ax = plt.subplots()

# ax.plot(
#     dfmmodel.external_forcings.boundaries['BC_flow_in']['time'],
#     dfmmodel.external_forcings.boundaries['BC_flow_in']['value'],
#     label='Discharge [m3/s]'
# )

# ax.plot(
#     dfmmodel.external_forcings.boundaries['BC_wlev_down']['time'],
#     dfmmodel.external_forcings.boundaries['BC_wlev_down']['value'],
#     label='Water level [m+NAP]'
# )

# ax.set_ylabel('Value (discharge or waterlevel)')
# ax.set_xlabel('Time [minutes]')

# ax.legend();


# ### Initial conditions

# There are four ways to set the initial conditions. First, global water level or depth can be set. In the example, we use a global water depth of 0.5 m, but we could also use the equivalent function "set_initial_waterlevel".

# Initial water depth is set to 0.5 m
dfmmodel.external_forcings.set_initial_waterdepth(0.5)


# It is also possible to define a certain area, using a polygon, with alternative initial conditions (level or depth).

# init_special = gpd.read_file(data_path+'/GIS/init_waterlevel_special.shp')
# dfmmodel.external_forcings.set_initial_waterlevel(10.0, polygon=init_special.geometry[0], name='test_polygon')


# ### Lateral flow

# Lateral flow can be obtained from the coupling with the RR-model, or by providing time series. Here, these are read from a Sobek model. In the coupling below, nodes that are not linked to a RR-boundary node are assumed to have a prescribed time series.
# 
# If a DFM-model is run offline, timeseries should be provided for all laterals.

###For adding the lateral inflow we import SOBEK results. To do so we use hkvsobekpy. For more info on this module, see: https://github.com/HKV-products-services/hkvsobekpy
# # Add the lateral inflows also from the SOBEK results. Naote that the column names in the his-file need to match
# # the id's of the imported lateral locations at the top of this notebook.
# rehis = hkvsobekpy.read_his.ReadMetadata(data_path+'/sobekdata/QLAT.HIS', hia_file='auto')
# param = [p for p in rehis.GetParameters() if 'disch' in p][0]
# lateral_discharge = rehis.DataFrame().loc[:, param]
# lateral_discharge.drop('lat_986', inplace=True, axis=1)

# dfmmodel.external_forcings.io.read_laterals(hydamo.laterals, lateral_discharges=lateral_discharge)


# ### Observation points

# Observation points are now written in the new format, where once can discriminate between 1D ('1d') and 2D ('2d') observation points. This can be done using the optional argument 'locationTypes'. If it is omitted, all points are assumed to be 1d. 1D-points are always snapped to a the nearest branch. 2D-observation points are always defined by their X/Y-coordinates.
# 
# Note: add_points can be called only once: once dfmodel.observation_points is filled,the add_points-method is not available anymore. Observation point coordinates can be definied eiher as an (x,y)-tuple or as a shapely Point-object.


# from shapely.geometry import Point
# dfmmodel.observation_points.add_points([Point((141150, 393700)),(141155, 393705),Point((145155, 394705)),(145150, 394700)],['ObsPt1','ObsPt2','ObsPt2D1','ObsPt2D2'], locationTypes=['1d','1d','2d','2d'])


# ### Settings and writing
# 
# Finally, we adjust some settings and export the coupled FM-RR model. For more info on the settings: https://content.oss.deltares.nl/delft3d/manuals/D-Flow_FM_User_Manual.pdf
# 
# The 1D/2D model (FM) is written to the sub-folder 'fm'; RR-files are written to 'rr'. An XML-file (dimr-config.xml) describes the coupling between the two. Note that both the GUI and Interaktor do not (yet) support RR, so the only way to carry out a coupled simulation is using DIMR.
# 


# Runtime and output settings
# for FM model
dfmmodel.mdu_parameters['refdate'] = 16720101
dfmmodel.mdu_parameters['tstart'] = 0.0 * 3600
dfmmodel.mdu_parameters['tstop'] = 24.0 * 5 * 3600
dfmmodel.mdu_parameters['hisinterval'] = '120. 0. 0.'
dfmmodel.mdu_parameters['cflmax'] = 0.7

# Create writer
dfmmodel.dimr_path = dimr_path
fm_writer = DFlowFMWriter(dfmmodel, output_dir=output_path, name='1672')

# Write as model
fm_writer.objects_to_ldb()
fm_writer.write_all()


# Finished!



