"""Lightcurve JSON I/O (no heapy dependency — safe for unit tests)."""

import json
import os
import shutil

import numpy as np

from .constants import LC_ARTIFACTS


def persist_lc_files(temp_dir, dest_dir):
    """Move heapy lightcurve artifacts from temp_dir to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    for fname in os.listdir(temp_dir):
        if fname in LC_ARTIFACTS:
            shutil.move(os.path.join(temp_dir, fname), os.path.join(dest_dir, fname))
    for sub in ('pgsignal',):
        src = os.path.join(temp_dir, sub)
        if os.path.isdir(src):
            dst = os.path.join(dest_dir, sub)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)


def mirror_lc_dir(src_dir, dst_dir):
    """Copy LC artifacts (files + pgsignal/) to canonical detector path."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        s = os.path.join(src_dir, fname)
        d = os.path.join(dst_dir, fname)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def infer_bin_edges_from_centers(t):
    """Infer histogram edges from bin centers (for step plots)."""
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return np.array([])
    if len(t) == 1:
        half = 0.25
        return np.array([t[0] - half, t[0] + half])
    inner = (t[:-1] + t[1:]) / 2.0
    left = t[0] - (inner[0] - t[0])
    right = t[-1] + (t[-1] - inner[-1])
    return np.concatenate([[left], inner, [right]])


def traces_from_lc_json(data):
    """Build plot trace dicts from a bb_lc.json / rebin_lc.json payload."""
    traces = []
    edges = np.array(data.get('edges', []), dtype=float)
    offset = data.get('time_offset')
    for tr in data['traces']:
        x = np.array(tr['x'], dtype=float)
        if offset is not None:
            x = x - float(offset)
        trace = {
            'name': tr['name'], 'type': 'scatter',
            'x': x,
            'y': np.array(tr['y'], dtype=float),
        }
        if len(edges) > 1:
            e = edges.copy()
            if offset is not None:
                e = e - float(offset)
            trace['edges'] = e
        if tr.get('error_y'):
            trace['error_y'] = np.array(tr['error_y'], dtype=float)
        traces.append(trace)
    return traces


def write_lc_json(path, payload):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f)


def load_lc_json(path):
    with open(path) as f:
        return json.load(f)


def bb_payload_from_arrays(edges, re_binsize, re_cts, re_bcts, p0, time_offset=None):
    edges = np.asarray(edges, dtype=float)
    re_binsize = np.asarray(re_binsize, dtype=float)
    re_cts = np.asarray(re_cts, dtype=float)
    re_bcts = np.asarray(re_bcts, dtype=float)
    t = (edges[:-1] + edges[1:]) / 2.0
    src_rate = re_cts / re_binsize
    bkg_rate = re_bcts / re_binsize
    src_err = np.sqrt(np.maximum(re_cts, 1.0)) / re_binsize
    bkg_err = np.sqrt(np.maximum(re_bcts, 1.0)) / re_binsize
    payload = {
        'method': 'heapy.pgSignal.bblock',
        'p0': float(p0),
        'edges': edges.tolist(),
        'traces': [
            {'name': 'source lightcurve', 'x': t.tolist(), 'y': src_rate.tolist(),
             'error_y': src_err.tolist()},
            {'name': 'background lightcurve', 'x': t.tolist(), 'y': bkg_rate.tolist(),
             'error_y': bkg_err.tolist()},
        ],
    }
    if time_offset is not None:
        payload['time_offset'] = float(time_offset)
        payload['edges_rel'] = (edges - float(time_offset)).tolist()
    return payload


def rebin_payload_from_tte_arrays(bins, t, reexps, src, bkg, src_err, bkg_err,
                                  time_offset=None):
    bins = np.asarray(bins, dtype=float)
    edges = np.unique(bins.flatten())
    payload = {
        'method': 'heapy.extract_rebin_curve',
        'edges': edges.tolist(),
        'traces': [
            {'name': 'source lightcurve', 'x': np.asarray(t, dtype=float).tolist(),
             'y': (np.asarray(src, dtype=float) / reexps).tolist(),
             'error_y': (np.asarray(src_err, dtype=float) / reexps).tolist()},
            {'name': 'background lightcurve', 'x': np.asarray(t, dtype=float).tolist(),
             'y': (np.asarray(bkg, dtype=float) / reexps).tolist(),
             'error_y': (np.asarray(bkg_err, dtype=float) / reexps).tolist()},
        ],
    }
    if time_offset is not None:
        payload['time_offset'] = float(time_offset)
        payload['edges_rel'] = (edges - float(time_offset)).tolist()
    return payload


def bb_traces_from_pgsignal(pgsignal_dir):
    """Rebuild BB LC traces from heapy ``pgsignal/*.json`` (fallback)."""
    with open(os.path.join(pgsignal_dir, 'block_res.json')) as f:
        block = json.load(f)
    with open(os.path.join(pgsignal_dir, 'snr_res.json')) as f:
        snr = json.load(f)
    edges = np.asarray(block['edges'], dtype=float)
    re_binsize = np.asarray(block['re_binsize'], dtype=float)
    re_cts = np.asarray(snr['re_cts'], dtype=float)
    re_bcts = np.asarray(snr['re_bcts'], dtype=float)
    t = (edges[:-1] + edges[1:]) / 2.0
    src_rate = re_cts / re_binsize
    bkg_rate = re_bcts / re_binsize
    src_err = np.sqrt(np.maximum(re_cts, 1.0)) / re_binsize
    bkg_err = np.sqrt(np.maximum(re_bcts, 1.0)) / re_binsize
    return [
        {'name': 'source lightcurve', 'type': 'scatter',
         'x': t, 'y': src_rate, 'error_y': src_err, 'edges': edges},
        {'name': 'background lightcurve', 'type': 'scatter',
         'x': t, 'y': bkg_rate, 'error_y': bkg_err, 'edges': edges},
    ]


def load_heapy_bayesian_blocks_lc(lc_dir):
    """Load Bayesian-block LC traces for plotting (``bb_lc.json`` or ``pgsignal/``)."""
    bb_path = os.path.join(lc_dir, 'bb_lc.json')
    if os.path.isfile(bb_path):
        return traces_from_lc_json(load_lc_json(bb_path))
    pg_dir = os.path.join(lc_dir, 'pgsignal')
    if os.path.isdir(pg_dir):
        return bb_traces_from_pgsignal(pg_dir)
    raise FileNotFoundError(f'No bb_lc.json or pgsignal/ in {lc_dir}')


def load_heapy_fixed_lc(lc_dir, parse_plotly_html=None):
    """Load uniform-bin source + background, trigger-relative."""
    json_path = os.path.join(lc_dir, 'lc_fixed.json')
    if os.path.isfile(json_path):
        return traces_from_lc_json(load_lc_json(json_path))
    if parse_plotly_html is None:
        raise FileNotFoundError(f'No lc_fixed.json in {lc_dir}')
    html_path = os.path.join(lc_dir, 'lc.html')
    if not os.path.isfile(html_path):
        raise FileNotFoundError(f'No lc_fixed.json or lc.html in {lc_dir}')
    traces = parse_plotly_html(html_path)
    offset = None
    for fname in ('lc_fixed.json', 'bb_lc.json', 'rebin_lc.json'):
        path = os.path.join(lc_dir, fname)
        if os.path.isfile(path):
            offset = load_lc_json(path).get('time_offset')
            if offset is not None:
                offset = float(offset)
                break
    if offset is None:
        return traces
    out = []
    for tr in traces:
        t = dict(tr)
        t['x'] = np.asarray(tr['x'], dtype=float) - offset
        if tr.get('edges') is not None:
            t['edges'] = np.asarray(tr['edges'], dtype=float) - offset
        out.append(t)
    return out


def load_heapy_rebin_lc(lc_dir, parse_plotly_html=None):
    """Load SNR-rebinned source + background traces for plotting."""
    json_path = os.path.join(lc_dir, 'rebin_lc.json')
    if os.path.isfile(json_path):
        return traces_from_lc_json(load_lc_json(json_path))

    if parse_plotly_html is None:
        raise FileNotFoundError(f'No rebin_lc.json in {lc_dir}')
    html_path = os.path.join(lc_dir, 'rebin_lc.html')
    fixed_path = os.path.join(lc_dir, 'lc.html')
    if not os.path.isfile(html_path) or not os.path.isfile(fixed_path):
        raise FileNotFoundError(
            f'No rebin_lc.json or rebin_lc.html+lc.html in {lc_dir}')

    net_tr = None
    for tr in parse_plotly_html(html_path):
        if tr.get('name', '').strip().lower() == 'net lightcurve':
            net_tr = tr
            break
    if net_tr is None:
        raise FileNotFoundError(f'No net lightcurve in {html_path}')

    bkg_tr = None
    for tr in parse_plotly_html(fixed_path):
        if 'background' in tr.get('name', '').lower():
            bkg_tr = tr
            break
    if bkg_tr is None:
        raise FileNotFoundError(f'No background in {fixed_path}')

    t = np.asarray(net_tr['x'], dtype=float)
    net_y = np.asarray(net_tr['y'], dtype=float)
    net_err = np.asarray(net_tr.get('error_y', np.sqrt(np.abs(net_y))), dtype=float)
    bkg_y = np.interp(t, bkg_tr['x'], bkg_tr['y'])
    bkg_err = np.interp(t, bkg_tr['x'], bkg_tr.get('error_y', np.sqrt(bkg_tr['y'])))
    src_y = net_y + bkg_y
    src_err = np.sqrt(net_err ** 2 + bkg_err ** 2)
    edges = infer_bin_edges_from_centers(t)
    return [
        {'name': 'source lightcurve', 'type': 'scatter',
         'x': t, 'y': src_y, 'error_y': src_err, 'edges': edges},
        {'name': 'background lightcurve', 'type': 'scatter',
         'x': t, 'y': bkg_y, 'error_y': bkg_err, 'edges': edges},
    ]


def combine_fixed_bin_traces(traces_by_det, time_offset=None):
    """Sum source/background rates from per-detector fixed-bin traces."""
    src_stack, bkg_stack, t_ref, edges_ref = [], [], None, None
    for traces in traces_by_det.values():
        src = next((tr for tr in traces if 'source' in tr['name'].lower()), None)
        bkg = next((tr for tr in traces if 'background' in tr['name'].lower()), None)
        if src is None:
            continue
        if t_ref is None:
            t_ref = np.asarray(src['x'], dtype=float)
            edges_ref = src.get('edges')
        src_y = np.interp(t_ref, src['x'], src['y'])
        src_stack.append(src_y)
        if bkg is not None:
            bkg_stack.append(np.interp(t_ref, bkg['x'], bkg['y']))
    if not src_stack:
        raise ValueError('No source traces to combine')
    src_sum = np.sum(src_stack, axis=0)
    bkg_sum = np.sum(bkg_stack, axis=0) if bkg_stack else np.zeros_like(src_sum)
    src_err = np.sqrt(np.maximum(src_sum, 0.0))
    payload = {
        'method': 'sum_selected_dets',
        'dets': list(traces_by_det),
        'edges': (np.asarray(edges_ref).tolist() if edges_ref is not None
                  else infer_bin_edges_from_centers(t_ref).tolist()),
        'traces': [
            {'name': 'source lightcurve', 'x': t_ref.tolist(), 'y': src_sum.tolist(),
             'error_y': src_err.tolist()},
            {'name': 'background lightcurve', 'x': t_ref.tolist(), 'y': bkg_sum.tolist(),
             'error_y': np.sqrt(np.maximum(bkg_sum, 0.0)).tolist()},
        ],
    }
    if time_offset is not None:
        payload['time_offset'] = float(time_offset)
        payload['edges_rel'] = (
            np.asarray(payload['edges'], dtype=float) - float(time_offset)).tolist()
    return payload
