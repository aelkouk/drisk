# Purpose:
# Record of revisions:
# Date 		Programmer 		Description of change
# ======== 	============= 	=====================
# 11/09/19 	A. Elkouk     	Original code

import numpy as np
import xarray as xr
from scipy import stats
import os
from mpi4py import MPI

from distribution import cal_smi

def ro_inputs(qtotpath):
    files = os.listdir(qtotpath)
    infiles = {}
    for fname in files:
        if ('historical' in fname) and ('qsb' not in fname):
            parts = fname.split('_')
            basename = '_'.join(parts[:2])
            tmp = []
            for fname2 in files:
                if (basename in fname2) and ('historical' not in fname2) and ('qsb' not in fname2):
                    tmp.append(fname2)
            infiles[fname] = tmp
    fn_rank = []
    for key, vals in infiles.items():
        key = (qtotpath + key, qtotpath + key.replace('qs', 'qsb'))
        fn_rank.append((key, key, True))
        for val in vals:
            val = (qtotpath + val, qtotpath + val.replace('qs', 'qsb'))
            fn_rank.append((key, val, False))
    return fn_rank

def dict_inputs(inpath):
    files = os.listdir(inpath)
    infiles = {}
    for fname in files:
        if 'historical' in fname:
            parts = fname.split('_')
            basename = '_'.join(parts[:2])
            tmp = []
            for fname2 in files:
                if (basename in fname2) and ('historical' not in fname2):
                    tmp.append(fname2)
            infiles[fname] = tmp
    return infiles

def sum_sm_depth(ds, hm):
    depths = {'clm45':[0.0175, 0.0276, 0.0455, 0.0750, 0.1236, 0.2038, 0.3360],
            'lpjml':[0.2, 0.3, 0.5],
            'orchidee':[0.001, 0.003, 0.006, 0.012, 0.023, 0.047, 0.094, 0.188, 0.375],
            'matsiro':[0.05, 0.2, 0.75],
            'jules-w1':[0.1, 0.25, 0.65]}
    if hm in depths.keys():
        if hm == 'orchidee':
            dsa = ds.sel(solay=slice(1, 9))
        elif hm == 'clm45':
            dsa = ds.sel(depth=slice(0, 7))
        elif hm == 'jules-w1':
            dsa = ds.sel(soil=slice(0.1, 0.65))
        elif hm == 'lpjml':
            dsa = ds.sel(depth=slice(0.1, 0.75))
        elif hm == 'matsiro':
            dsa = ds.sel(lev=slice(1, 3))
        z = np.array(depths[hm])
        sm_weighted = dsa['soilmoist'].values.transpose(0, 2, 3, 1) * (z / z.sum())
        sm = np.sum(sm_weighted, axis=-1)
    else:
        sm = ds['soilmoist'].values
    return sm

def mpi_calc_smi(hist_fpath, prj_fpath, usehist_flag=False, outpath='../../2_pipeline/SMI/'):
    ds = xr.open_dataset(hist_fpath, decode_times=False)
    ds_hist = ds.isel(time=slice(-420, -60))
    hm_nm = os.path.basename(hist_fpath).split('_')[0]
    sm_hist = sum_sm_depth(ds_hist, hm_nm)
    nt_hist, nlat, nlon = sm_hist.shape
    ngrd = nlat * nlon
    sm_histix = np.reshape(sm_hist, (nt_hist, ngrd))
    # sm_histix = np.ascontiguousarray(tmp, dtype=float)
    if usehist_flag:
        ds = ds_hist
        nt_prj = nt_hist
        sm_prjix = sm_histix
        outfname = os.path.basename(hist_fpath)
    else:
        ds = xr.open_dataset(prj_fpath, decode_times=False)
        sm_prj = sum_sm_depth(ds, hm_nm)
        nt_prj, _, _ = sm_prj.shape
        sm_prjix = np.reshape(sm_prj, (nt_prj, ngrd))
        # sm_prjix = np.ascontiguousarray(tmp, dtype=float)
        outfname = os.path.basename(prj_fpath)
    smi_prj = cal_smi(sm_histix, sm_prjix, np.nan, None, nt_prj, ngrd)
    smi_out = smi_prj#.reshape((nt_prj, nlat, nlon))
    dsout = xr.Dataset(data_vars={'SMI':(('time', 'ngrd'), smi_out)},
                       coords={'time': ds.time})
    # dsout = xr.Dataset(data_vars={'SMI':(('time', 'lat', 'lon'), smi_out)},
    #                    coords={'time': ds.time, 'lat': ds.lat, 'lon': ds.lon})
    dsout.to_netcdf(os.path.join(outpath, outfname), encoding={'SMI': {"dtype": "f4"}})
    print(outfname, 'DONE')

def calc_qtot(fpaths):
    ds_qs = xr.open_dataset(fpaths[0], decode_times=False)
    ds_qsb = xr.open_dataset(fpaths[1], decode_times=False)
    qtot = ds_qs['qs']+ds_qsb['qsb']
    return qtot

def mpi_calc_ri(hist_fpath, prj_fpath, usehist_flag=False, outpath='../../2_pipeline/SMI/'):
    ds_qtot = calc_qtot(hist_fpath)
    qtot_hist = ds_qtot.isel(time=slice(-420, -60))
    nt_hist, nlat, nlon = qtot_hist.shape
    ngrd = nlat * nlon
    qtot_histix = np.reshape(qtot_hist.values, (nt_hist, ngrd))
    if usehist_flag:
        ds_qtot = qtot_hist
        nt_prj = nt_hist
        qtot_prjix = qtot_histix
        outfname = os.path.basename(hist_fpath[0].replace('qs', 'qtot'))
    else:
        ds_qtot = calc_qtot(prj_fpath)
        nt_prj, _, _ = ds_qtot.shape
        qtot_prjix = np.reshape(ds_qtot.values, (nt_prj, ngrd))
        outfname = os.path.basename(prj_fpath[0].replace('qs', 'qtot'))
    ri_out = cal_smi(qtot_histix, qtot_prjix, np.nan, None, nt_prj, ngrd)
    dsout = xr.Dataset(data_vars={'RI':(('time', 'ngrd'), ri_out)},
                       coords={'time': ds_qtot.time})
    # dsout = xr.Dataset(data_vars={'SMI':(('time', 'lat', 'lon'), smi_out)},
    #                    coords={'time': ds.time, 'lat': ds.lat, 'lon': ds.lon})
    dsout.to_netcdf(os.path.join(outpath, outfname), encoding={'RI': {"dtype": "f4"}})
    print(outfname, 'DONE')

def calc_dpr_dths(infile, varname, ths, outpath):
    ds =  xr.open_dataset(infile, decode_times=False)
    di = ds[varname].values
    nt, ngrd = di.shape
    nth = ths.size
    if 'historical' in infile:
        di_ths = np.stack([di]*nth).transpose(1,2,0)
        dpr = (di_ths<=ths).sum(axis=0)/nt
        dsout = xr.Dataset(data_vars={'DPr':(('nth', 'ngrd'), dpr.T)})
    else:
        di_dts = np.stack([di[25*12:55*12], di[-30*12:]])
        di_ths = np.stack([di_dts] * nth).transpose(1, 2, 3, 0)
        dpr = (di_ths <= ths).sum(axis=1) / (30*12)
        dsout = xr.Dataset(data_vars={'DPr':(('nth', 'nT', 'ngrd'), dpr.transpose(2,0,1))})
    parts = os.path.basename(infile).split('_')
    outfname = '_'.join([parts[0], parts[1], parts[3], varname]) + '.nc'
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

def calc_exp_dths(infile, varname, pop, countries_ma, ncoun, ths, outpath):
    ds =  xr.open_dataset(infile, decode_times=False)
    di = ds[varname].values
    nt, ngrd = di.shape
    ny = nt//12
    di_ny = np.reshape(di, (ny, 12, ngrd)).transpose(1,0,2)
    nyp, npt, nssp, nlat, nlon = pop.shape
    dny = nyp-ny
    popr = np.reshape(pop[dny:], (ny, npt, nssp, ngrd)).transpose(1,2,0,3)
    nth = len(ths)

    dths_exp_ncoun = np.full((npt, nssp, nth, ny, ncoun), np.nan)
    dths_exp_glob = np.full((npt, nssp, nth, ny), np.nan)
    for i, thi in enumerate(ths):
        di_ny_thi = (di_ny <= thi).sum(axis=0)
        expthi = di_ny_thi * popr
        for ci in range(ncoun):
            ci_ma = (countries_ma == ci)
            expthi_ci = np.nansum(expthi[:,:,:, ci_ma], axis=-1)/12
            dths_exp_ncoun[:,:,i,:,ci] = expthi_ci
        expthi_grd = np.nansum(expthi, axis=-1)/12
        dths_exp_glob[:,:,i,:] = expthi_grd

    dsout = xr.Dataset(data_vars={'exp_cn':(('npt', 'nssp', 'nth', 'ny', 'ncn'), dths_exp_ncoun),
                                  'exp_glob':(('npt', 'nssp', 'nth', 'ny'), dths_exp_glob),})
    parts = os.path.basename(infile).split('_')
    outfname = '_'.join([parts[0],parts[1],parts[3],varname]) + '.nc'
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

def calc_exp_dths_cstclim(infile, varname, pop, countries_ma, ncoun, ths, outpath):
    ds =  xr.open_dataset(infile, decode_times=False)
    di = ds[varname].values
    nt, ngrd = di.shape
    ny, npt, nssp, nlat, nlon = pop.shape
    popr = np.reshape(pop, (ny, npt, nssp, ngrd)).transpose(1,2,0,3)
    nth = len(ths)

    dths_exp_ncoun = np.full((npt, nssp, nth, ny, ncoun), np.nan)
    dths_exp_glob = np.full((npt, nssp, nth, ny), np.nan)
    for i, thi in enumerate(ths):
        di_ny_thi = (di <= thi).sum(axis=0)
        expthi = di_ny_thi * popr
        for ci in range(ncoun):
            ci_ma = (countries_ma == ci)
            expthi_ci = np.nansum(expthi[:,:,:, ci_ma], axis=-1)/nt
            dths_exp_ncoun[:,:,i,:,ci] = expthi_ci
        expthi_grd = np.nansum(expthi, axis=-1)/nt
        dths_exp_glob[:,:,i,:] = expthi_grd

    dsout = xr.Dataset(data_vars={'exp_cn':(('npt', 'nssp', 'nth', 'ny', 'ncn'), dths_exp_ncoun),
                                  'exp_glob':(('npt', 'nssp', 'nth', 'ny'), dths_exp_glob),})
    parts = os.path.basename(infile).split('_')
    outfname = '_'.join([parts[0],parts[1],parts[3],varname]) + '.nc'
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

def calc_exp_dths_cstpop(infile, varname, pop, countries_ma, ncoun, ths, outpath):
    ds =  xr.open_dataset(infile, decode_times=False)
    di = ds[varname].values
    nt, ngrd = di.shape
    ny = nt//12
    di_ny = np.reshape(di, (ny, 12, ngrd)).transpose(1,0,2)
    nyp, npt, nlat, nlon = pop.shape
    dny = nyp-ny
    popr = np.reshape(pop[dny:], (ny, npt, ngrd)).transpose(1,0,2)
    nth = len(ths)

    dths_exp_ncoun = np.full((npt, nth, ny, ncoun), np.nan)
    dths_exp_glob = np.full((npt, nth, ny), np.nan)
    for i, thi in enumerate(ths):
        di_ny_thi = (di_ny <= thi).sum(axis=0)
        expthi = di_ny_thi * popr
        for ci in range(ncoun):
            ci_ma = (countries_ma == ci)
            expthi_ci = np.nansum(expthi[:, :, ci_ma], axis=-1)/12
            dths_exp_ncoun[:, i,:,ci] = expthi_ci
        expthi_grd = np.nansum(expthi, axis=-1)/12
        dths_exp_glob[:, i] = expthi_grd

    dsout = xr.Dataset(data_vars={'exp_cn':(('npt', 'nth', 'ny', 'ncn'), dths_exp_ncoun),
                                  'exp_glob':(('npt', 'nth', 'ny'), dths_exp_glob),})
    parts = os.path.basename(infile).split('_')
    outfname = '_'.join([parts[0],parts[1],parts[3],varname]) + '.nc'
    dsout.to_netcdf(os.path.join(outpath, outfname))
    print(outfname, 'DONE')

def lognormal_population(pop, refpop):
    ''' Normalize population using log10-normal cdf'''
    logdata = np.log(refpop[refpop>0])
    params = (np.nanmean(logdata), np.nanstd(logdata))
    popcdf = stats.norm.cdf(np.log(pop), *params)
    #popref_cdf = stats.norm.cdf(np.log(refpop), *params)
    return popcdf#, popref_cdf

def calc_exp_uncer_glob(infile, varname, pop, ths):
    ds =  xr.open_dataset(infile, decode_times=False)
    di = ds[varname].values
    nt, ngrd = di.shape
    ny = nt//12
    di_ny = np.reshape(di, (ny, 12, ngrd)).transpose(1,0,2)
    nyp, npt, nssp, nlat, nlon = pop.shape
    dny = nyp-ny
    popr = np.reshape(pop[dny:], (ny, npt, nssp, ngrd)).transpose(1,2,0,3)
    nth = len(ths)
    
    dths_exp_glob = np.full((npt, nssp, nth, ny, ngrd), np.nan)
    for i, thi in enumerate(ths):
        di_ny_thi = (di_ny <= thi).sum(axis=0)/12
        expthi = di_ny_thi * popr
        #expthi_grd = np.nansum(expthi, axis=-1) / 12
        dths_exp_glob[:,:,i,:] = expthi
    return dths_exp_glob