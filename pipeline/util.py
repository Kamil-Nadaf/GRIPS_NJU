"""Small filesystem helpers used by notebooks."""

import json
import os
import shutil

import h5py
import pandas as pd

from .constants import DATA_BASE
from .paths import read_meta


def get_bayspec_path(grb_name, data_base=None):
    data_base = data_base or DATA_BASE
    return os.path.join(
        data_base, grb_name, 'data/tresolved/bayspec', f'{grb_name}_bayspec_data.h5')


def view_hdf5(filepath, key=None):
    with h5py.File(filepath, 'r') as f:
        available_keys = list(f.keys())
        print(f'Available keys: {available_keys}\n')
    if key and key in available_keys:
        return pd.read_hdf(filepath, key=key)
    if key:
        print(f"ERROR: Key '{key}' not found")


def clean_dir(path, prefix=None):
    if not os.path.exists(path):
        return
    for item in os.listdir(path):
        if prefix and not item.startswith(prefix):
            continue
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)
    print(f'Cleaned: {path}' + (f" (prefix='{prefix}')" if prefix else ''))


def show_directory_tree(path, max_depth=3, max_files=5):
    import subprocess

    print(f'Current working directory: {os.getcwd()}\n')
    print(f'=== Directory Tree: {path} ===\n')
    try:
        result = subprocess.run(
            ['tree', '-L', str(max_depth), path],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
            return
    except Exception:
        pass

    def print_tree(current_path, prefix='', depth=0):
        if depth >= max_depth:
            return
        try:
            items = sorted(os.listdir(current_path))
            dirs = [i for i in items if os.path.isdir(os.path.join(current_path, i))]
            files = [i for i in items if os.path.isfile(os.path.join(current_path, i))]
            for i, d in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                connector = '└── ' if is_last_dir else '├── '
                print(f'{prefix}{connector}{d}/')
                new_prefix = prefix + ('    ' if is_last_dir else '│   ')
                print_tree(os.path.join(current_path, d), new_prefix, depth + 1)
            files_to_show = files[:max_files]
            remaining = len(files) - max_files
            for i, f in enumerate(files_to_show):
                is_last = (i == len(files_to_show) - 1) and remaining <= 0
                connector = '└── ' if is_last else '├── '
                print(f'{prefix}{connector}{f}')
            if remaining > 0:
                print(f'{prefix}└── ... ({remaining} more files)')
        except PermissionError:
            print(f'{prefix}[Permission Denied]')

    print(f'{os.path.basename(path) or path}/')
    print_tree(path)
    print()


def list_grb_results(grb_name, data_base=None):
    data_base = data_base or DATA_BASE
    h5_path = get_bayspec_path(grb_name, data_base=data_base)
    print(f'\n=== {grb_name} results ===')
    if os.path.isfile(h5_path):
        view_hdf5(h5_path)
    else:
        print(f'  No HDF5 at {h5_path}')
    for label, root in [
        ('tint heapy', os.path.join(data_base, grb_name, 'data/tintegrated/heapy')),
        ('tres heapy', os.path.join(data_base, grb_name, 'data/tresolved/heapy')),
        ('tint bayspec', os.path.join(data_base, grb_name, 'data/tintegrated/bayspec')),
        ('tres bayspec', os.path.join(data_base, grb_name, 'data/tresolved/bayspec')),
    ]:
        if not os.path.isdir(root):
            continue
        print(f'\n{label}: {root}')
        versions = os.path.join(root, 'versions')
        if os.path.isdir(versions):
            print(f'  versions: {sorted(os.listdir(versions))}')
        meta = read_meta(os.path.join(root, 'pipeline_meta.json'))
        if meta:
            print(f'  meta: {json.dumps(meta, indent=2, default=str)}')
