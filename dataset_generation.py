from xgcm import Grid
import pop_tools
import gcsfs
import fsspec as fs
import numpy as np
import xesmf as xe
import xarray as xr
import random
import matplotlib.pyplot as plt
import warnings
from xgcm import Grid
import gcm_filters
import aerobulk
from aerobulk import noskin

def load_and_combine_cm26(
    filesystem: gcsfs.GCSFileSystem, inline_array=False
) -> xr.Dataset:
    """Loads, combines, and preprocesses CM2.6 data
    Steps:
    - Interpolate ocean velocities on ocean tracer points (with xgcm)
    - Regrid atmospheric variables to ocean tracer grid (with xesmf)
    - Match time and merge datasets
    - Adjust units for aerobulk input
    - Calculate relative wind components
    """
    kwargs = dict(
        consolidated=True, use_cftime=True, inline_array=inline_array, engine="zarr"
    )
    print("Load Data")
    mapper = filesystem.get_mapper("gs://cmip6/GFDL_CM2_6/control/surface")
    ds_ocean = xr.open_dataset(mapper, chunks={"time": 3}, **kwargs)
    mapper = filesystem.get_mapper("gs://cmip6/GFDL_CM2_6/control/ocean_boundary")
    xr.open_dataset(mapper, chunks={"time": 3}, **kwargs)

    # xarray says not to do this
    # ds_atmos = xr.open_zarr('gs://cmip6/GFDL_CM2_6/control/atmos_daily.zarr', chunks={'time':1}, **kwargs) # noqa: E501
    mapper = filesystem.get_mapper("gs://cmip6/GFDL_CM2_6/control/atmos_daily.zarr")
    ds_atmos = xr.open_dataset(mapper, chunks={"time": 120}, **kwargs).chunk(
        {"time": 3}
    )
    mapper = filesystem.get_mapper("gs://cmip6/GFDL_CM2_6/grid")
    ds_oc_grid = xr.open_dataset(mapper, chunks={}, **kwargs)
    # ds_oc_grid  = cat["GFDL_CM2_6_grid"].to_dask()

    print("Align in time")
    # cut to same time
    all_dims = set(list(ds_ocean.dims) + list(ds_atmos.dims))
    ds_ocean, ds_atmos = xr.align(
        ds_ocean,
        ds_atmos,
        join="inner",
        exclude=(di for di in all_dims if di != "time"),
    )

    print("Interpolating ocean velocities")
    # interpolate ocean velocities onto the tracer points using xgcm
    from xgcm import Grid

    # add xgcm comodo attrs
    ds_ocean["xu_ocean"].attrs["axis"] = "X"
    ds_ocean["xt_ocean"].attrs["axis"] = "X"
    ds_ocean["xu_ocean"].attrs["c_grid_axis_shift"] = 0.5
    ds_ocean["xt_ocean"].attrs["c_grid_axis_shift"] = 0.0
    ds_ocean["yu_ocean"].attrs["axis"] = "Y"
    ds_ocean["yt_ocean"].attrs["axis"] = "Y"
    ds_ocean["yu_ocean"].attrs["c_grid_axis_shift"] = 0.5
    ds_ocean["yt_ocean"].attrs["c_grid_axis_shift"] = 0.0
    grid = Grid(ds_ocean)
    # fill missing values with 0, then interpolate.
    sst_wet_mask = ~np.isnan(ds_ocean["surface_temp"])
    # TODO: Maybe stencil out the nans from SST? This is done again in aerobulk-python
    ds_ocean["u_ocean"] = grid.interp_like(
        ds_ocean["usurf"].fillna(0), ds_ocean["surface_temp"]
    ).where(sst_wet_mask)
    ds_ocean["v_ocean"] = grid.interp_like(
        ds_ocean["vsurf"].fillna(0), ds_ocean["surface_temp"]
    ).where(sst_wet_mask)
    # FIXME: All of these additional operations (filling/masking) add lot more tasks...

    print("Regrid Atmospheric Data")
    # Start regridding the atmosphere onto the ocean grid
    # Load precalculated regridder weights from group bucket
    # TODO: Maybe this should be an input argument?
    path = "gs://leap-persistent/jbusecke/scale-aware-air-sea/regridding_weights/CM26_atmos2ocean.zarr"  # noqa: E501
    mapper = filesystem.get_mapper(path)
    ds_regridder = xr.open_zarr(mapper).load()
    regridder = xe.Regridder(
        ds_atmos.olr.to_dataset(name="dummy")
        .isel(time=0)
        .reset_coords(drop=True),  # this is the same dumb problem I keep having with
        ds_ocean.surface_temp.to_dataset(name="dummy")
        .isel(time=0)
        .reset_coords(drop=True),
        "bilinear",
        weights=ds_regridder,
        periodic=True,
    )
    ds_atmos_regridded = regridder(
        ds_atmos[["slp", "v_ref", "u_ref", "t_ref", "q_ref", "wind", "rh_ref"]]
    )  # We are only doing noskin for now , 'swdn_sfc', 'lwdn_sfc'

    ## combine into merged dataset
    ds_merged = xr.merge(
        [
            ds_atmos_regridded,
            ds_ocean[["surface_temp", "u_ocean", "v_ocean"]],
        ]
    )
    print("Modify units")
    # ds_merged = ds_merged.transpose(
    #     'xt_ocean', 'yt_ocean', 'time'
    # )
    # fix units for aerobulk
    ds_merged["surface_temp"] = ds_merged["surface_temp"] + 273.15
    ds_merged["slp"] = ds_merged["slp"] * 100  # check this

    print("Mask nans")
    # atmos missing values are filled with 0s, which causes issues with the filtering
    # Ideally this should be masked before the regridding, but xesmf fills with 0 again...
    mask = ~np.isnan(ds_merged['surface_temp'].isel(time=0).reset_coords(drop=True))
    for mask_var in ['slp', 't_ref', 'q_ref', 'v_ref', 'u_ref', 'wind', 'rh_ref']:
        ds_merged[mask_var] = ds_merged[mask_var].where(mask)
    
    
    # also apply this mask to certain coordinates from the grid dataset (for now only tracer_area since that 
    for mask_coord in ['area_t']:
        ds_merged.coords[mask_coord] = ds_oc_grid[mask_coord].where(mask,0.0).astype(np.float64)
    # The casting to float64 is needed to avoid that weird bug where the manual global weighted ave
    # is not close to the xarray weighted mean (I was not able to reproduce this with an example)
    
    # Ideally this should be masked before the regridding,
    # but xesmf fills with 0 again...
    mask = ~np.isnan(ds_merged["surface_temp"])
    for mask_var in ["slp", "t_ref", "q_ref"]:
        ds_merged[mask_var] = ds_merged[mask_var].where(mask)

    # Calculate relative wind
    print("Calculate relative wind")
    ds_merged["u_relative"] = ds_merged["u_ref"] - ds_merged["u_ocean"]
    ds_merged["v_relative"] = ds_merged["v_ref"] - ds_merged["v_ocean"]

    return ds_merged

def preprocess_data():
    
    warnings.filterwarnings("ignore")
    
    # Loading ocean data
    kwargs = dict(consolidated=True, use_cftime=True, engine="zarr")
    ocean_path = "gs://cmip6/GFDL_CM2_6/control/surface"
    ds_ocean = xr.open_dataset(fs.get_mapper(ocean_path), chunks={"time": 3}, **kwargs)
    ocean_boundary_path = "gs://cmip6/GFDL_CM2_6/control/ocean_boundary"
    ds_ocean_boundary = xr.open_dataset(fs.get_mapper(ocean_boundary_path), chunks={"time": 3}, **kwargs)
    grid_path = "gs://cmip6/GFDL_CM2_6/grid"
    ds_ocean_grid = xr.open_dataset(fs.get_mapper(grid_path), chunks={}, **kwargs)
    
    # combine all dataset on the ocean grid together
    ds_ocean = xr.merge([ds_ocean_grid, ds_ocean, ds_ocean_boundary], compat='override')
    
    # interpolate ocean velocities onto the tracer points using xgcm
    ds_ocean["xu_ocean"].attrs["axis"] = "X"
    ds_ocean["xt_ocean"].attrs["axis"] = "X"
    ds_ocean["xu_ocean"].attrs["c_grid_axis_shift"] = 0.5
    ds_ocean["xt_ocean"].attrs["c_grid_axis_shift"] = 0.0
    ds_ocean["yu_ocean"].attrs["axis"] = "Y"
    ds_ocean["yt_ocean"].attrs["axis"] = "Y"
    ds_ocean["yu_ocean"].attrs["c_grid_axis_shift"] = 0.5
    ds_ocean["yt_ocean"].attrs["c_grid_axis_shift"] = 0.0
    grid = Grid(ds_ocean)

    # fill missing values with 0, then interpolate.
    tracer_ref = ds_ocean["surface_temp"]
    sst_wet_mask = ~np.isnan(tracer_ref)

    ds_ocean["u_ocean"] = grid.interp_like(
        ds_ocean["usurf"].fillna(0), tracer_ref
    ).where(sst_wet_mask)
    ds_ocean["v_ocean"] = grid.interp_like(
        ds_ocean["vsurf"].fillna(0), tracer_ref
    ).where(sst_wet_mask)

    # Loading atmos data
    atmos_path = "gs://cmip6/GFDL_CM2_6/control/atmos_daily.zarr"
    ds_atmos = xr.open_dataset(fs.get_mapper(atmos_path), chunks={"time": 120}, **kwargs).chunk(
        {"time": 3}
    )
    # rename the atmos data coordinates only to CESM conventions
    ds_atmos = ds_atmos.rename({'grid_xt':'lon', 'grid_yt':'lat'})

    ds_ocean["surface_temp"] = ds_ocean["surface_temp"] + 273.15
    ds_atmos["slp"] = ds_atmos["slp"] * 100  # TODO: Double check this

    # Merge atmos and ocean datasets
    ds_merged  = load_and_combine_cm26(fs, inline_array=True)
    
    return ds_merged

def filter_inputs(
    da: xr.DataArray,
    wet_mask: xr.DataArray,
    dims: list,
    filter_scale: int,
    filter_type: str = "taper",
) -> xr.DataArray:
    """filters input using gcm-filters"""
    if filter_type == "gaussian":
        input_filter = gcm_filters.Filter(
            filter_scale=filter_scale,
            dx_min=1,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
            grid_vars={"wet_mask": wet_mask},
        )
    elif filter_type == "taper":
        # transition_width = np.pi/2
        transition_width = np.pi
        input_filter = gcm_filters.Filter(
            filter_scale=filter_scale,
            transition_width=transition_width,
            dx_min=1,
            filter_shape=gcm_filters.FilterShape.TAPER,
            grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
            grid_vars={"wet_mask": wet_mask},
        )
    elif filter_type == "tripolar_pop":
        input_filter = gcm_filters.Filter(
            filter_scale=filter_scale,
            dx_min=1,
            filter_shape=gcm_filters.FilterShape.GAUSSIAN,
            grid_type=gcm_filters.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED,
            grid_vars={"area":da.TAREA,"wet_mask": wet_mask},
        )

    else:
        raise ValueError(
            f"`filter_type` needs to be `gaussian` or `taper', got {filter_type}"
        )
    out = input_filter.apply(da, dims=dims)
    out.attrs["filter_scale"] = filter_scale
    out.attrs["filter_type"] = filter_type
    return out

def filter_inputs_dataset(
    ds: xr.Dataset,
    dims: list,
    filter_scale: int,
    timedim: str = "time",
    filter_type: str = "gaussian", #taper
    t_range: tuple = None  # expects (start_idx, end_idx)
) -> xr.Dataset:
    """Wrapper that filters a whole dataset, generating a wet_mask from
    the nanmask of the first timestep (if time is present)."""
    ds_out = xr.Dataset()

    # create a wet mask that only allows values which are 'wet' in all variables

    wet_masks = []
    for var in ds.data_vars:
        da = ds[var]
        if timedim in da.dims:
            mask_da = da.isel({timedim: 0})
        else:
            mask_da = da
        wet_masks.append((~np.isnan(mask_da)))

    combined_wet_mask = xr.concat(wet_masks, dim="var").all("var").astype(int)

    for var in ds.data_vars:
        ds_out[var] = filter_inputs(
            ds[var], combined_wet_mask, dims, filter_scale, filter_type=filter_type
        )

    ds_out.attrs["filter_scale"] = filter_scale
    ds_out.attrs["filter_type"] = filter_type
    
    # Apply time slicing if requested
    if t_range is not None:
        start, end = t_range
        ds_out = ds_out.isel({timedim: slice(start, end)})
        
    return ds_out

def compute_heatflux(
    ds_hr,
    ds_lr,
    algo='ecmwf',
    zt=2,
    zu=10,
    dims=['yt_ocean', 'xt_ocean'],
    filter_scale=50,
    filter_type='gaussian'
):
    """
    Compute surface turbulent heat fluxes (ql + qh) from high-res and low-res datasets,
    apply spatial filtering, and return HR, LR, and difference.

    Parameters:
        ds_hr (xr.Dataset): High-resolution dataset
        ds_lr (xr.Dataset): Low-resolution dataset
        algo (str): Algorithm for aerobulk (e.g., 'ecmwf', 'coare')
        zt (int): Reference height for temperature
        zu (int): Reference height for wind
        dims (list): Dimensions to apply filter on (default: ['yt_ocean', 'xt_ocean'])
        filter_scale (int): Length scale for gcm-filters
        filter_type (str): Filter type ('gaussian', 'taper', etc.)

    Returns:
        heatflux_hr (xr.DataArray): High-res latent + sensible heat flux
        heatflux_lr (xr.DataArray): Low-res latent + sensible heat flux
        heatflux_diff (xr.DataArray): heatflux_hr - heatflux_lr (both smoothed)
    """
    # High-res flux
    ql_hr, qh_hr, *_ = noskin(
        ds_hr.surface_temp,
        ds_hr.t_ref,
        ds_hr.q_ref,
        ds_hr.u_relative,
        ds_hr.v_relative,
        slp=ds_hr.slp,
        algo=algo,
        zt=zt,
        zu=zu,
    )
    heatflux_hr = qh_hr + ql_hr
    heatflux_hr_smooth = filter_inputs_dataset(
        heatflux_hr.to_dataset(name="heatflux"),
        dims=dims,
        filter_scale=filter_scale,
        filter_type=filter_type
    )["heatflux"]

    # Low-res flux
    ql_lr, qh_lr, *_ = noskin(
        ds_lr.surface_temp,
        ds_lr.t_ref,
        ds_lr.q_ref,
        ds_lr.u_relative,
        ds_lr.v_relative,
        slp=ds_lr.slp,
        algo=algo,
        zt=zt,
        zu=zu,
    )
    heatflux_lr = qh_lr + ql_lr
    heatflux_lr_smooth = filter_inputs_dataset(
        heatflux_lr.to_dataset(name="heatflux"),
        dims=dims,
        filter_scale=filter_scale,
        filter_type=filter_type
    )["heatflux"]

    # Difference
    heatflux_diff = heatflux_hr_smooth - heatflux_lr_smooth

    return xr.Dataset({
        "heatflux_hr": heatflux_hr_smooth,
        "heatflux_lr": heatflux_lr_smooth,
        "heatflux_diff": heatflux_diff
    })