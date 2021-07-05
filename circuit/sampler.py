import argparse
import os
import sys
from subprocess import call

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sampler.py',
        description='sample threat value for approximation',
        add_help=True
    )
    parser.add_argument('--depth', dest='depth', default='5', type=str)
    parser.add_argument('-n', dest='num', default='100000', type=str)
    parser.add_argument('-p', dest='per', default='10000', type=str)
    parser.add_argument('--beta', dest='beta', default='0.999', type=str)
    parser.add_argument('--out', dest='out',
                        default='data', type=str)
    args = parser.parse_args()

    print('total', args.num, 'data /', args.per, ' samples per (s, a)')

    call(['./circuit/sampler', args.depth, args.num, args.per, args.beta])

    ans = np.loadtxt('output.dat', dtype=np.float32)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'circuit_threat.npy'), 'wb') as f:
        np.save(f, ans)
        f.close()

    print('completed.')
