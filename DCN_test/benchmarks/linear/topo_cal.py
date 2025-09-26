import os
import sys
import math
import argparse

import numpy as np
from tqdm import tqdm


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Generate raw_topo/<topo_name>.txt by running COUDER per TM')
    parser.add_argument('--topo_name', required=True, help='Base name; read raw_data/topo_<topo_name>.csv')
    parser.add_argument('--output_dir', default=os.path.join('..', '..', 'raw_topo'), help='Directory to write output .txt')
    parser.add_argument('--s_capacity', type=float, default=1.0, help='Per-link single-port bandwidth (capacity per single line)')
    parser.add_argument('--r_limit', type=int, default=3, help='Per-pod port-count upper bound (sum_j n[i,j] <= r_limit)')
    parser.add_argument('--integer', action='store_true', default=True, help='Use integer mode in COUDER')
    return parser.parse_args(argv)


def load_csv_iter(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support comma or whitespace separated values
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            yield [float(x) for x in parts]


def infer_n_and_validate(length):
    n_float = math.sqrt(length)
    n = int(n_float)
    if n * n != length:
        raise ValueError(f'Row length {length} is not a perfect square; cannot infer NxN TM')
    return n


def main(argv):
    args = parse_args(argv)

    # Import COUDER from tools with a stable relative sys.path
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from tools.COUDER import COUDER

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.topo_name}.txt")

    input_csv = os.path.join(repo_root, 'raw_data', f"{args.topo_name}.csv")
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    it = load_csv_iter(input_csv)
    first_row = None
    try:
        first_row = next(it)
    except StopIteration:
        raise RuntimeError('Input CSV is empty')

    N = infer_n_and_validate(len(first_row))
    # r_limit = N-1
    # Build constant s_matrix and R_c
    s_matrix = np.full((N, N), float(args.s_capacity), dtype=float)
    R_c = np.full((N,), int(args.r_limit), dtype=int)

    # Process first row then the rest
    # Count total non-empty lines for progress bar
    total_lines = 0
    with open(input_csv, 'r', encoding='utf-8') as _f:
        for _line in _f:
            if _line.strip():
                total_lines += 1

    with open(output_path, 'w', encoding='utf-8') as out_f:
        def process_row(row_vals):
            d_wave = np.array(row_vals, dtype=float).reshape((N, N))
            # Ensure self-demand is zero
            for i in range(N):
                d_wave[i, i] = 0.0
            tag = 'INTEGER' if args.integer else 'RELAXED'
            _, _, n_value = COUDER(s_matrix, R_c, d_wave, N, tag=tag)
            # import pdb;pdb.set_trace()
            # n_value is an NxN numpy array
            flat = n_value.astype(int).reshape(-1)
            out_f.write(','.join(map(str, flat)))
            out_f.write('\n')

        process_row(first_row)
        with tqdm(total=total_lines, initial=1, desc='COUDER per TM', unit='tm') as pbar:
            for row in it:
                if len(row) != N * N:
                    raise ValueError(f'Inconsistent row length {len(row)}; expected {N*N}')
                process_row(row)
                pbar.update(1)

    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main(sys.argv[1:])


