"""SAA, visibility, detector angles, skymap."""

import os

from .constants import ALL_DETS


def extract_geometry(ctx, dets=None):
    """SAA passage, visibility, detector angles, skymap — per gbm.ipynb."""
    from heapy.geos.geometry import gbmGeometry

    if dets is None:
        dets = ALL_DETS

    geo_dir = ctx.paths.geometry
    os.makedirs(geo_dir, exist_ok=True)

    gbm_geo = gbmGeometry(file=ctx.gbm_rtv.rtv_res['poshist'])
    print('saa_passage:', gbm_geo.saa_passage)
    print('location_visible:', gbm_geo.get_location_visible(
        ra=ctx.ra, dec=ctx.dec,
        met=[ctx.fermi_met - 500, ctx.fermi_met, ctx.fermi_met + 500]))

    for det in dets:
        angle = gbm_geo.get_detector_angle(
            ra=ctx.ra, dec=ctx.dec, det=det,
            met=[ctx.fermi_met - 100, ctx.fermi_met, ctx.fermi_met + 100])
        print(det, angle)

    gbm_geo.extract_skymap(ra=ctx.ra, dec=ctx.dec, met=ctx.fermi_met, savepath=geo_dir)
    print(f'Geometry saved to {geo_dir}')
    return geo_dir
