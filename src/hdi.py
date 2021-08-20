# Purpose: Normalized population vulnerability from 1-HDI (HDI, human development index)
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 04/2021 	A. Elkouk     	Original code


import json
import pandas as pd
import scipy.stats as st

from rasterio import features
from affine import Affine

def prepare_HDI(fpath_HDIssps, fpath_HDIunpd):


    # HDI SSPs
    df_ssp = pd.read_csv(fpath_HDIssps)
    index = [(s.split(' - ')[0], int(s.split()[-1])) for s in df_ssp['obs']]
    multidx = pd.MultiIndex.from_tuples(index, names=['country', 'year'])
    df_hdi = df_ssp.iloc[:, 1:].set_index(multidx)
    df_hdi.dropna(how='all', inplace=True)

    # HDIunpd
    with open(fpath_HDIunpd, 'r') as myfile:
        data = myfile.read()
    # parse file
    hdi_un = json.loads(data)

    # Correction of country names in HDIssps to match HDIunpd
    corrections = {
                    'Bolivia':'Bolivia (Plurinational State of)',
                    'Cape Verde':'Cabo Verde',
                    'China, Hong Kong SAR':'Hong Kong, China (SAR)',
                    'Czech Republic':'Czechia',
                    'Democratic Republic of the Congo':'Congo (Democratic Republic of the)',
                    'Republic of Korea':'Korea (Republic of)',
                    'Republic of Moldova':'Moldova (Republic of)',
                    'Swaziland':'Eswatini (Kingdom of)',
                    'TFYR Macedonia':'North Macedonia',
                    'United Republic of Tanzania':'Tanzania (United Republic of)',
                    'United States of America':'United States',
                    }
    #
    names = df_hdi.index.get_level_values('country')
    cnames = np.unique(names).tolist()
    for ci in cnames:
        if ci in corrections.keys():
            cnames[cnames.index(ci)] = corrections[ci]
    hdi_un15 = np.zeros(len(cnames))
    codes_cnames = {}
    for i, ci in enumerate(cnames):
        for key,value in hdi_un['country_name'].items():
            if value == ci:
                hdi_un15[i] = hdi_un['indicator_value'][key]['137506']['2015']
                codes_cnames[ci] = key

    # Normalize HDI SSPs: Vulnerability Index (VI)
    inv_hdi_un15 = 1 - hdi_un15
    inv_hdi_ssps = 1 - df_hdi.values.ravel()
    iqr = np.percentile(inv_hdi_un15, 75) - np.percentile(inv_hdi_un15, 25)
    bw = 0.9 * min(iqr, np.std(inv_hdi_un15)) * (inv_hdi_un15.size) ** (-0.2)
    outer = (inv_hdi_ssps[:, None] - inv_hdi_un15) / bw
    cdf = st.norm.cdf(outer, *(0, 1)).mean(axis=1)
    cdf = cdf.reshape(df_hdi.values.shape)
    outer = (inv_hdi_un15[:, None] - inv_hdi_un15) / bw
    cdf15 = st.norm.cdf(outer, *(0, 1)).mean(axis=1)

    # Dataframe for VI
    tmp = df_hdi.index.get_level_values('country')
    allcn = tmp.tolist()
    for ci in allcn:
        if ci in corrections.keys():
            allcn[allcn.index(ci)] = corrections[ci]
    tmp = df_hdi.index.get_level_values('year').tolist()
    index = pd.MultiIndex.from_arrays([allcn, tmp], names=['country', 'year'])
    df_vi = pd.DataFrame(cdf, columns=df_hdi.columns, index=index)
    df_vi_nonnorm = pd.DataFrame(1 - df_hdi.values, columns=df_hdi.columns, index=index)
    df_hdi = pd.DataFrame(df_hdi.values, columns=df_hdi.columns, index=index)
    for i, ci in enumerate(cn):
        df_vi.loc[ci, 'HDI_2015'] = np.array([cdf15[i]] * 14)
        df_vi_nonnorm.loc[ci, 'HDI_2015'] = np.array([inv_hdi_un15[i]] * 14)
    return df_vi, df_vi_nonnorm

shpfile = '../../0_data/shpfiles/countries/ne_10m_admin_0_countries_lakes.shp'

def HDIdf_to_grid(lons, lats, shpfile, df_vi, df_vi_nonnorm):
    nlat, nlon = lats.size, lons.size
    # Convert VI for NAS countries into grid
    trans = Affine.translation(lons[0], lats[0])
    scale = Affine.scale(lons[1] - lons[0], lats[1] - lats[0])
    transform = trans * scale
    out_shape = (nlat, nlon)

    vi = np.full((14,6,nlat,nlon), np.nan)
    vi_nonnorm = np.full((14,6,nlat,nlon), np.nan)

    reader = cartopy.io.shapereader.Reader(shpfile)
    for i, record in enumerate(reader.records()):
        iso_code = record.attributes['ADM0_A3']
        for ci,iso in codes_cn.items():
            if iso_code==iso:
                geom = record.geometry
                vals = df_vi.loc[ci].values.ravel()
                tmp = np.full((vals.size,nlat,nlon), np.nan)
                vals_nonnorm = df_vi_nonnorm.loc[ci].values.ravel()
                tmp_nonnorm = np.full((vals_nonnorm.size,nlat,nlon), np.nan)
                for i in range(vals.size):
                    shapes = [(geom, vals[i])]
                    tmp[i] = features.rasterize(shapes, out_shape=out_shape, fill=np.nan,
                                           transform=transform, dtype=float,
                                           all_touched=True)
                    shapes = [(geom, vals_nonnorm[i])]
                    tmp_nonnorm[i] = features.rasterize(shapes, out_shape=out_shape, fill=np.nan,
                                           transform=transform, dtype=float,
                                           all_touched=True)
                tmp = np.reshape(tmp, vi.shape)
                vi[~np.isnan(tmp)] = tmp[~np.isnan(tmp)]
                tmp_nonnorm = np.reshape(tmp_nonnorm, vi.shape)
                vi_nonnorm[~np.isnan(tmp_nonnorm)] = tmp_nonnorm[~np.isnan(tmp_nonnorm)]

    dsout = xr.DataArray(data=vi, attrs={'long_name': '1-HDI normalized globaly based on UN 1-HDI in 2015'})
    dsout.to_netcdf(os.path.join(outpath, outfname_vi))
    dsout = xr.DataArray(data=vi_nonnorm, attrs={'long_name': '1-HDI non-normalized globaly'})
    dsout.to_netcdf(os.path.join(outpath, outfname_vinonnorm))

    return vi, vi_nonnorm
