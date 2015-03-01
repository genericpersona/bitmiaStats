#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
import pickle
import random
from string import hexdigits
import sys
import zlib

class EV0LottoSim(object):
    def __init__(self, num_trials, odds, jackpot, tix_per_trial, nons=3, verbose=False):
        '''
        Create a simulation for the EV0 lotto scheme to see
        if a self-sustaining lotto is feasible.

        Params
        ------
            num_trials: int
                number of trials to run

            odds: int
                the odds of winning the lotto in the form 1 in n odds
                odds should be set to the value of n

                for example, 1 in 100 odds would have pass in 100

            jackpot: float
                BTC making up the jackpot
                jackpot = ticket price * odds  

            tix_per_trial: int
                number of tickets to generate for each trial

            price_threshold: float
                how many times the initial price to go before
                changing the odds

                for example, 5 means once the ticket price is
                5x or more the initial price, the odds change
                to reset the price back to the original

            percent_booster: float 
                percentage of the ticket sales to roll over into
                a booster fund

                defaults to 1.0 as a fixed jackpot is being used

            nons: int
                Number of numbers in the drawing

                defaults to 3

            verbose: bool
                whether or not to be verbose is output during the
                running of the trials

                without verbosity, only final results are printed

                with it, messages are printed to stderr during each
                trial run

                defaults to False 
        '''
        # Boring saving of parameters
        self.jackpot = jackpot
        self.nons = nons
        self.num_trials = num_trials
        self.odds = odds
        self.ticket_price = round(jackpot / odds, 8)
        self.tix_per_trial = tix_per_trial
        self.verbose = verbose

        # Helpful renaming
        self.rand = random.SystemRandom()
        self.se = sys.stderr

        # Save some of the accounting variables
        self.gain = 0.0
        self.loss = jackpot # Must shclep out money for first jackpot
        self.num_neg_trials = 1 # Start out w/ gain - loss < 0
        self.num_wins = 0
        self.stat_attrs = ('gain', 'loss', 'jackpot', 'num_neg_trials', \
                            'num_wins', 'odds', 'ticket_price')
        self.stats = {k: [] for k in self.stat_attrs}

        # Save the initial values of each parameter
        self.init = {k: getattr(self, k) for k in self.stat_attrs}
        if self.verbose:
            self.se.write('{}\n'.format(self.init))

        # Load some data for the simulation
        self._loadBlockHashes()
        self._winningNumbersGen()

    def _generateTickets(self):
        '''
        Generate random tickets to compare against the blockchain
        for determining a winner.
        '''
        tickets = set() 

        for n in range(self.tix_per_trial):
            hds = [self.rand.choice(hexdigits).upper() \
                    for _ in range(self.nons)]
            tickets.add(''.join(sorted(hds)))

        return tickets

    def _loadBlockHashes(self):
        '''
        Load the block hash data from disk
        '''
        with open('bh_hist.pickle', 'rb') as f:
            self.hash_list = pickle.loads(zlib.decompress(f.read()))

    def _reportResults(self):
        '''
        Report results on running the trials.

        Not meant to be called outside of the run method
        '''
        header = '\nSimulation results for {} trials'.format(self.num_trials)
        print(header)
        print('{}\n'.format('-' * len(header)))

        for attr in self.stat_attrs:
            print('{}: {}'.format(attr, getattr(self, attr)))

        print('Total Gain: {}'.format(round(self.gain - self.loss, 8)))

    def _winningNumbersGen(self):
        '''
        Save a lucky numbers generator in the winning_nums attribute
        '''
        wns = [''.join(sorted(hds[-self.nons:])) for hds in self.hash_list]
        self.winning_nums = itertools.cycle(wns)

    def run(self):
        '''
        Run the simulation and report on it afterwards.

        In each trial:

            -- Generate the required number of tickets
            -- Use the actual blockchain to get the winning numbers
            -- See if any ticket was a winner
            -- Adjust the balance of the lotto accordingly
        '''
        for n in range(self.num_trials):
            # Each trial brings in a certain amount of ticket sales
            ticket_sales = self.ticket_price * self.tix_per_trial
            self.gain += ticket_sales 

            # See if there's a winner
            if next(self.winning_nums) in self._generateTickets():
                # We have a winner!
                self.num_wins += 1

                # Pay out the jackpot
                self.loss += self.jackpot

                # Start the new jackpot
                self.loss += self.jackpot

            # See if we're operating negatiely
            if self.gain - self.loss < 0:
                self.num_neg_trials += 1

            # Save all values in the stats dictionary
            if self.verbose:
                header = 'Trial {} Results'.format(n + 1)
                self.se.write('{}\n'.format(header))
                self.se.write('{}\n'.format('-' * len(header)))

            for attr in self.stat_attrs:
                self.stats[attr].append(getattr(self, attr))

                if self.verbose:
                    self.se.write('  {}: {}\n'.format(attr, getattr(self, attr)))

            # Make some space b/w trial reports to stderr
            if self.verbose and n != self.num_trials - 1:
                self.se.write('\n')

        self._reportResults()

if __name__ == '__main__':
    desc = '''
           Run bitmia EV0 Lotto Simulations
           '''
    p = argparse.ArgumentParser(description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog='bitmia EV0 lotto simulator')

    p.add_argument( '-d'
                  , '--num-digits'
                  , help='Number of hex digits in lotto'
                  , default=2
                  , type=int
                  )

    p.add_argument( '-j'
                  , '--jackpot'
                  , default=1.0
                  , help='Starting jackpot in BTC'
                  , type=float
                  )

    p.add_argument( '-n'
                  , '--num-trials'
                  , default=10**5
                  , help='Number of trials to run'
                  , type=int
                  )
                  
    p.add_argument( '-o'
                  , '--odds'
                  , default=136
                  , help='Odds of winning as a 1 in <odds> chance'
                  , type=int
                  )

    p.add_argument( '-t'
                  , '--tix-per-trial'
                  , default=10
                  , help='Tickets generated per trial'
                  , type=int
                  )

    p.add_argument( '-v'
                  , '--verbose'
                  , action='store_true'
                  , help='Control verbosity of output'
                  )

    args = p.parse_args()

    sim = EV0LottoSim(args.num_trials, args.odds,
            args.jackpot, args.tix_per_trial,
            args.num_digits, args.verbose)
    sim.run()
