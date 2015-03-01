#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Imports
import argparse
from collections import Counter
import cPickle as pickle
import itertools as it
import json
import multiprocessing as mp
import os
import pprint
import random
from string import hexdigits
import subprocess as sp
import zlib

import matplotlib.pyplot as plt

# Globals
ARGS = None

class BitcoinCLI(object):
    def __init__(self, **kwargs):
        '''
        Create a subprocess calling bitcoin-cli client.

        Each kwarg is based on anything used on the command line
        for bitcoin-cli.

        Can be called as follows:

            btc_cli = BitcoinCLI(conf='btc.conf')
            print int(btc_cli('getblockcount'))
            print btc_cli('getblockhash', int(btc_cli('getblockcount')))
        '''
        self.args = ['-{}={}'.format(k, v) for k, v in kwargs.items()]
        self.cmnd = ['bitcoin-cli'] + self.args
        self._helpCommands()

    def _helpCommands(self):
        '''
        Save a list of RPC commands in the attribute
        commands.
        '''
        lines = sp.check_output(self.cmnd + ['help']).split('\n')
        cmnds = map(lambda s: s.split()[0], # Take the first word
                        # bool gets non-empty lines
                        # islower captures all the commands
                        filter(str.islower, filter(bool, lines)))
        self.commands = cmnds
    
    def __call__(self, *args):
        '''
        '''
        cmnd = self.cmnd + [args[0]] + map(json.dumps, args[1:])
        return sp.check_output(cmnd).strip()

def block_hash(block):
    '''
    Get the rightmost n hex digits of a given
    block's hash.
    '''
    bh = BitcoinCLI(conf=ARGS.config)('getblockhash', block)
    if ARGS.verbose:
        print 'block {}: {}'.format(block, bh)
    return bh

def block_hash_list(hash_list=[]):
    '''
    Generate the distribution of the n rightmost
    hex digit strings in every block hash in the
    mainnet block chain.

    Returns a Counter object.

    Params
    ------
        hash_list: Counter object 
            If not [], update with just new block hashes
    '''
    last_block = 0 if not hash_list else len(hash_list)
    btc_cli = BitcoinCLI(conf=ARGS.config)
    total_blocks = int(btc_cli('getblockcount'))

    pool = mp.Pool(processes=mp.cpu_count()*2)
    new_hash_list = hash_list + \
                        pool.map(block_hash, range(last_block, total_blocks+1))

    if hash_list:
        with open(ARGS.output + '.pickle', 'wb') as f:
            f.write(zlib.compress(pickle.dumps(new_hash_list)))

    return new_hash_list

def build_parser():
    '''
    Build the command line parser for the histogram maker
    '''
    desc = '''
           Run block hash histograms
           '''
    p = argparse.ArgumentParser(description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog='bitmia block hash histogram generator')

    p.add_argument( '-c'
                  , '--config'
                  , default='bitcoin.conf'
                  , help='Config file to pass to bitcoin-cli'
                  )
    p.add_argument( '-n'
                  , '--num-digits'
                  , help='Number of hex digits in lotto'
                  , default=3
                  , type=int
                  )
    p.add_argument( '-o'
                  , '--output'
                  , default='bh_hist'
                  , help='Output file for histogram data (w/o extension)'
                  )
    p.add_argument( '-v'
                  , '--verbose'
                  , action='store_true'
                  , help='Control verbosity of output'
                  )

    return p

def hash_hist_report(hash_list):
    '''
    Reports on the results of the n-digit hashes found 
    in the blockchain.

    Saves a gnuplot output in the file specified by the
    output parameter.
    '''
    def get_n_hds(hds):
        hdss = hds[-n:]
        hdss = ''.join(sorted(hdss))
        return hdss.upper()
    n = ARGS.num_digits
    hash_hist = Counter(get_n_hds(hds) for hds in hash_list)
    hds = filter(lambda d: not d.islower(), hexdigits)

    # Print out some brief stats
    print 'Number of {} hex digit strings with replacement: {}'.format(\
            n, len(list(it.combinations_with_replacement(hds, n))))
    print 'Number of {} hex digit strings in blockchain: {}'.format(\
            n, len(hash_hist.keys()))
    print 'Most common strings:\n{}'.format(\
            pprint.pformat(hash_hist.most_common()))

    # Make a histogram
    plt.bar(range(len(hash_hist.values())), 
            list(hash_hist.values()), 
            align='center')
    plt.xticks(range(len(hash_hist.keys())), list(hash_hist.keys()),
            rotation='vertical')
    plt.show()

def main():
    '''
    Parse args and do stuff.
    '''
    global ARGS 
    args = build_parser().parse_args()
    ARGS = args

    if os.path.exists(args.output + '.pickle'):
        with open(args.output + '.pickle', 'rb') as f:
            hash_list = pickle.loads(zlib.decompress(f.read()))

        hash_list = block_hash_list(hash_list)
    else:
        hash_list = block_hash_list()

        # Save a pickled version to disk
        with open(args.output + '.pickle', 'wb') as f:
            f.write(zlib.compress(pickle.dumps(hash_list)))

    hash_hist_report(hash_list)

if __name__ == '__main__':
    main()
