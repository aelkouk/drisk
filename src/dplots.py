# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 06/2021 	A. Elkouk     	Original code


# Scratch

# import xarray as xr
# import numpy as np
# import pandas as pd
# import os
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy
# import matplotlib as mpl
# import seaborn as sns
# import string
# from itertools import cycle
# from shapely import geometry, ops
#
# def dpr_plot(ddpr, outpath):
#     mpl.rcParams['mathtext.fontset'] = 'stix'
#     mpl.rcParams['hatch.linewidth'] = 0.25
#     levels = np.arange(-40, 40.1, 5)
#     cm = ['#20183B',
#           '#377CC0',
#           '#4CC4DC',
#           '#ACC8D0',
#           # '#241C24',
#           '#F3EA8C',
#           '#FCCC0C',
#           '#F05824',
#           '#F2739B',
#           '#DA2424']
#     cmap = mpl.colors.LinearSegmentedColormap.from_list('cm', cm, N=levels.size)
#     dcnames = ['D0 (Abnormally-Dry)', 'D1 (Moderate-Drought)',
#                'D2 (Severe-Drought)', 'D3 (Extreme-Drought)',
#                'D4 (Exceptional-Drought)']
#     fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, dpi=100, figsize=(10, 14),
#                              gridspec_kw={'wspace': 0.1, 'hspace': 0.05},
#                              subplot_kw={'projection': ccrs.PlateCarree(), 'frameon': False}, )
#     ss = 2
#     Ti = 1
#     for i, thi in enumerate([0, 1, 2, 3, 4]):
#         ys = dpr_emean[di, ss, thi, Ti].reshape(nlat, nlon) * 100
#         z = (dpr_eagg[di, ss, thi, Ti].reshape(nlat, nlon) >= 66) * 1
#         ys[ys == 0] = np.nan
#         z = np.where(np.isnan(ys), np.nan, z)
#         ax = axes[i, di]
#         ax.set_title(titles[di][thi], fontsize=14)
#         ax.coastlines(linewidth=0.5, resolution="110m")
#         cs = ax.contourf(lons, lats, ys, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
#         ax.contourf(lons, lats, z, levels=[-0.9, .9, 1.2], hatches=['', '\\\\\\\\\\\\', ''], colors='none',
#                     alpha=0.)
#         ax.background_patch.set_visible(False)
#         ax.outline_patch.set_visible(False)
#         ax.set_ylim(-60, 95)
#
#     cbar = fig.colorbar(cs, ax=axes, orientation='horizontal', pad=0.02, shrink=0.8, drawedges=True, aspect=40,
#                         extendfrac='auto', spacing='uniform', )
#     cbar.ax.tick_params(axis='x', labelsize=12)
#     cbar.ax.set_xlabel('(%)', fontsize=12)
#     label_axes(axes.ravel(), loc=(0., 1.05), fontsize=14)
#     fig.savefig(outpath+'fig_dpr.jpeg', bbox_inches='tight', dpi=300)
#
# basepath = '/storage/elkoukah/empirical/'
# os.chdir(basepath)
# hist_fpath = '0_data/ISIMIP2b_soilmoist/clm45_gfdl-esm2m_ewembi_historical_2005soc_co2_soilmoist_global_monthly_1861_2005.nc4'
# prj_fpath = '0_data/ISIMIP2b_soilmoist/clm45_gfdl-esm2m_ewembi_rcp85_2005soc_co2_soilmoist_global_monthly_2006_2099.nc'
# outpath = '2_pipeline/drisk/tmp/'
# index_flag = 'SMI'
# hm, gcm, _, clim = os.path.basename(prj_fpath).split('_')[:4]
# Thist = slice(-360, None)
#
# ths = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
# outfname_dpr = '_'.join((hm, gcm, clim, index_flag + '.nc'))
# outfname_dpr_ref = '_'.join((hm, gcm, 'historical', index_flag + '.nc'))
# Ti = slice(-360, None)
# Ti_ref = slice(-360, None)
#
# ds_pop = xr.load_dataarray('0_data/POP/pop_total_rural.nc')
# pop = ds_pop.values
# ds_countries = xr.load_dataarray('0_data/COUNTRIES/countries.nc')
# countries_ma = ds_countries.values.ravel()
# ncoun = len(ds_countries.ISO)
# tp = 1
# yi = 6
# yi_ref = 0
# outfname_exp = '_'.join((hm,gcm,clim,index_flag,'exposure.nc'))
# outfname_exp_ref = '_'.join((hm,gcm,'historical',index_flag,'exposure.nc'))
# outfname_exp_cstclim = '_'.join((hm,gcm,'historical',index_flag,'exposure_cstclim.nc'))
# outfname_exp_cstpop = '_'.join((hm,gcm,clim,index_flag,'exposure_cstpop.nc'))
#
# outfname_expnorm = '_'.join((hm,gcm,clim,index_flag,'exposure_norm.nc'))
# outfname_expnorm_cstclim = '_'.join((hm,gcm,'historical',index_flag,'exposure_norm_cstclim.nc'))
# outfname_expnorm_cstpop = '_'.join((hm,gcm,clim,index_flag,'exposure_norm_cstpop.nc'))
#
# ds_hdi = xr.load_dataarray('0_data/POP/vi_norm.nc')
# hdi_norm = ds_hdi.values[:, :-1]
# hdi_norm_ref = ds_hdi.values[:, [-1]*5] # replicate hdiref 5 times for consistency with SSPs1-5
# yi_hdi = 12
# yi_hdi_ref = 0
# outfname_diri = '_'.join((hm,gcm,clim,index_flag,'diri.nc'))
# outfname_diri_cstclim = '_'.join((hm,gcm,'historical',index_flag,'diri_cstclim.nc'))
# outfname_diri_cstpop = '_'.join((hm,gcm,clim,index_flag,'diri_cstpop.nc'))
#
# rclass = np.arange(0., 0.91, 0.1)
# outfname_expdiri = '_'.join((hm,gcm,clim,index_flag,'expdiri.nc'))
# outfname_expdiri_cstclim = '_'.join((hm,gcm,'historical',index_flag,'expdiri_cstclim.nc'))
# outfname_expdiri_cstpop = '_'.join((hm,gcm,clim,index_flag,'expdiri_cstpop.nc'))
