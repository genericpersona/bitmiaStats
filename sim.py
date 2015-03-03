#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import itertools
from math import ceil
import pickle
import random
from string import hexdigits
import sys
import zlib

try:
    from gmpy2 import comb
except ImportError:
    quit('You must install gmpy2')

class EV0LottoSim(object):
    def __init__(self, num_trials, odds, jackpot, tix_per_trial, 
                       nons, rollover, perms, verbose=False):
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

            nons: int
                Number of numbers in the drawing

            rollover: float
                number b/w 0 and 1 representing the percentage of
                ticket sales to rollover into the next jackpot

            perms: bool
                whether to use permutations or combinations in
                checking for a winner

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
        self.perms = perms
        self.rollover = rollover
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
                           'num_wins', 'odds', 'ticket_price', 'tix_per_trial')
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
            if self.perms:
                tickets.add(''.join(hds))
            else:
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

        print('Max Jackpot: {}'.format(sorted(self.stats['jackpot'])[-1]))
        print('Total Gain: {}'.format(round(self.gain - self.loss, 8)))

    def _winningNumbersGen(self):
        '''
        Save a lucky numbers generator in the winning_nums attribute
        '''
        if self.perms:
            wns = [''.join(hds[-self.nons:]) for hds in self.hash_list]
        else:
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

                self.jackpot = self.init['jackpot']

            else:
                self.jackpot += ticket_sales * self.rollover

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

def at_least(trials, odds, wins):
    '''
    Calculates the probability of at least wins successes
    in n trials with a given odds of success.

    Params
    ------
        trials: int
            Number of trials

        odds: int
            Odds of success as denominator in 1 in odds
            expressions

        wins: int
            Minimum number of wins before negative net gain
    '''
    b_part = partial(bernoulli, trials, odds)
    p = map(b_part, range(wins, trials+1))
    return sum(p)

def bernoulli(trials, odds, wins):
    '''
    Bernoulli probability with n trials and 1 / odds
    chance of success.

    Return float b/w 0 and 1

    Params
    ------
        trials: int
            Number of trials

        odds: int
            Odds of success as denominator in 1 in odds
            expressions

        wins: int
            Minimum number of wins before negative net gain
    '''
    return comb(trials, wins) * \
           (1 / odds)**wins * \
           (1 - (1 / odds))**(trials - wins)

def loss_threshold(odds, jackpot, trials):
    '''
    Return the minimum number of wins needed before reaching
    negative net gain.

    Params
    ------
        odds: int
            the odds as passed into the simulator
            e.g., for 1 in 136 odds, pass in 136

        jackpot: float
            starting jackpot in BTC

        trials: int
            number of tickets sold

    '''
    max_loot = trials * float(jackpot / odds)
    min_wins = ceil(max_loot / jackpot)
    return int(min_wins)

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

    p.add_argument( '-p'
                  , '--perms'
                  , action='store_true'
                  , help='Whether order matters for winning the lotto'
                  )

    p.add_argument( '-r'
                  , '--rollover'
                  , default=0.0
                  , help='Percentage of ticket sales to roll into next jackpot'
                  , type=float
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

    sim = EV0LottoSim(args.num_trials, 
            args.odds,
            args.jackpot, 
            args.tix_per_trial,
            args.num_digits, 
            args.rollover,
            args.perms,
            args.verbose)
    sim.run()
