# Geometry scripts

These scripts take binary data of RyR locations and creates a h5 file representing a 2x2x2 micron geometry. The geometry consists of the following domains
 
- Cytosol
- Cleft space
- Junctional SR
- Non-junctional SR
- t-tubules

In addition the placement of RyR is included in the geometry. SERCA is not explicitly placed, but assumed to be active at the entire cytosol-SR interface. No LTCC are included in the geometry.

### Making a geometry file from a *ppm file

To produce a .h5 file from a .ppm file of RyR locations, use the `generate_cru_geometry_file` module. It can either be used as an import module in a script, or from the command line. Run 
```
python generate_cru_geometry_file --help
```
For a full list of options. For better control over the geometry creation process, take a look at the `geometry_util` module.

### Visualizing at the geometry file

The `visualize_geometry` module helps you visualize the resulting geometry stored in a .h5 file. It can also be used via command line arguments. Run
```
python visualize_geometry --help
```
For a full list of options

### To-do

- Update `generate_parameters` module (Started, but not finished)
- Update `single_run` module
- Update `read_case` module
- Update `get_stat` module
