#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
import itertools as it
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

try:
    import matplotlib.pyplot as plt
except ImportError:
    quit('What are you doing without matplotlib?!')

try:
    import numpy as np
except ImportError:
    quit('Get NumPy now, silly!')

class EV0LottoSim(object):
    def __init__(self, args):
        '''
        Create a simulation for the EV0 lotto scheme to see
        if a self-sustaining lotto is feasible.

        Params
        ------
            jackpot: float
                BTC making up the jackpot
                jackpot = ticket price * odds  

            tix_percentage: float
                percentage of EV0 ticket price to charge

            num_drawings: int
                total number of drawings

            tix_total: int
                total number of tickets to generate

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
        for k, v in vars(args).items():
            setattr(self, k, v)
        if self.perms:
            self.odds = it.permutations(hexdigits[:16], 
                                        self.num_digits)
        else:
            self.odds = it.combinations_with_replacement(hexdigits[:16], 
                                                         self.num_digits)
        self.odds = len(list(self.odds))
        self._setTicketPrice()

        # Helpful renaming
        self.rand = random.SystemRandom()
        self.se = sys.stderr

        # Save some of the accounting variables
        self.gain = 0.0
        self.loss = self.jackpot # Must shclep out money for first jackpot
        self.num_neg_trials = 0 # Haven't sold any tickets yet
        self.num_wins = 0
        self.stat_attrs = ('gain', 'loss', 'jackpot', 'num_neg_trials', \
                           'num_wins', 'odds', 'ticket_price', 'tix_percentage',
                           'tix_total')
        self.stats = {k: [] for k in self.stat_attrs}

        # Save the initial values of each parameter
        self.init = {k: getattr(self, k) for k in self.stat_attrs}
        if self.verbose:
            self.se.write('{}\n'.format(self.init))

        # Load some data for the simulation
        self._loadBlockHashes()
        self._winningNumbersGen()

    def _generateTicket(self):
        '''
        Generate random ticket to compare against the blockchain
        for determining a winner.
        '''
        hds = ''.join(set(hexdigits.upper()))
        ticket = [self.rand.choice(hds) for _ in range(self.num_digits)]
        
        if self.perms:
            return ''.join(ticket)
        else:
            return ''.join(sorted(ticket))

    def _generateTickets(self, n):
        '''
        Generate n random tickets as a set
        '''
        tickets = set(self._generateTicket() for _ in range(n))

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
        header = '\nSimulation results for {} drawings with {} tickets each'.\
                    format(self.num_drawings, self.tix_total)
        print(header)
        print('{}\n'.format('-' * len(header)))

        for attr in self.stat_attrs:
            print('{}: {}'.format(attr, getattr(self, attr)))

        print('Max Jackpot: {}'.format(sorted(self.stats['jackpot'])[-1]))
        print('Total Gain: {}'.format(round(self.gain - self.loss, 8)))

    def _setTicketPrice(self):
        '''
        Set the ticket price
        '''
        tp = self.jackpot / (self.odds * self.odds_reduction)
        self.ticket_price = round(tp, 8)

    def _winningNumbersGen(self):
        '''
        Save a lucky numbers generator in the winning_nums attribute
        '''
        nons = self.num_digits
        if self.perms:
            wns = [''.join(hds[-nons:]) for hds in self.hash_list]
        else:
            wns = [''.join(sorted(hds[-nons:])) for hds in self.hash_list]
        self.winning_nums = it.cycle(wns)

    def run(self):
        '''
        Run the simulation and report on it afterwards.

        In each trial:

            -- Generate the required number of tickets
            -- Use the actual blockchain to get the winning numbers
            -- See if any ticket was a winner
            -- Adjust the balance of the lotto accordingly
        '''
        for n in range(self.num_drawings):
            # Each trial brings in a gain from the ticket sold
            ticket_sales = self.ticket_price * self.tix_total
            self.gain += ticket_sales

            # See if there's a winner
            if next(self.winning_nums) in self._generateTickets(self.tix_total):
                # We have a winner!
                self.num_wins += 1

                # Pay out the jackpot
                self.loss += self.jackpot 

                # Reset the jackpot to its initial amount
                self.jackpot = self.init['jackpot']

                # Start the new jackpot
                self.loss += self.jackpot
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

            # Reset the ticket price if necessary
            self._setTicketPrice()

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

def plot_runs(args):
    '''
    Show the net gain and the time between wins on a plot
    '''
    def plot_net_gain():
        plt.plot(range(1, len(net_gain)+1),
                net_gain, 'gs')
        pos = len(list(filter(lambda n: n >=0, net_gain)))
        per_pos = round((pos / len(net_gain)) * 100, 2)
        plt.ylabel('BTC')
        plt.xlabel('Simulation #')
        plt.title('Net Gain ({}% Positive) (Max: {}) (Min: {})'.\
            format(per_pos, round(max(net_gain), 2), round(min(net_gain), 2)))

    def plot_tbws():
        plt.plot(range(1, len(tbws)+1),
                tbws, 'ro')
        plt.xlabel('Sample #')
        plt.ylabel('# Tickets b/w Wins')
        mean = round(np.mean(tbws), 2)
        std = round(np.std(tbws), 2)
        med = round(np.median(tbws), 2)
        title = 'Time Between Wins (Avg: {}) (Median: {}) (Std Dev: {})'.\
                format(mean, med, std)
        plt.title(title)

    net_gain = []
    tbws     = [] # Time between wins  
    for n in range(args.simulations):
        sim = run_sim(args)

        net_gain.append(sim.gain - sim.loss)

        # last_i contains the last index where a win occurred
        last_i = 0
        last_nw = 0
        for i, num_wins in enumerate(sim.stats['num_wins']):
            if num_wins != last_nw:
                tbws.append(i - last_i)
                last_i, last_nw = i, num_wins

    if args.net_gain:
        plot_net_gain()
    elif args.time_between_wins:
        plot_tbws()

    descr = '{:,} Simulations ({:,} Drawings) ({:,} Tickets Each) ({}% rollover)\n'.\
        format(args.simulations, args.num_drawings,
                args.tix_total, round(args.rollover * 100, 2))
    descr += '1 in {} Odds ({} BTC Jackpot)\n'.format(sim.odds, args.jackpot)
    plt.annotate(descr, (0,0), (0, -20), 
            xycoords='axes fraction', 
            textcoords='offset points', 
            va='top')
    if args.output:
        plt.savefig(args.output, format='png', dpi=600)
    plt.show()

def run_sim(args):
    '''
    Run a simulation and return the simulation 
    object
    '''
    sim = EV0LottoSim(args)
    sim.run()
    return sim

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
                  , '--num-drawings'
                  , default=1
                  , help='Number of drawings to hold'
                  , type=int
                  )

    p.add_argument( '--net-gain'
                  , action='store_true'
                  , help='Whether to plot net gain'
                  )

    p.add_argument( '-o'
                  , '--output'
                  , help='File path to save figure to'
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

    p.add_argument( '-s'
                  , '--simulations'
                  , help='Number of simulations to run'
                  , default=1
                  , type=int
                  )

    p.add_argument( '-t'
                  , '--tix-total'
                  , default=10
                  , help='Tickets generated per trial'
                  , type=int
                  )

    p.add_argument( '--time-between-wins'
                  , action='store_true'
                  , help='Whether to plot time between wins'
                  )

    p.add_argument( '-u'
                  , '--odds-reduction'
                  , default=1.0
                  , help='Percent to reduce odds for calculating ticket price'
                  , type=float
                  )

    p.add_argument( '-v'
                  , '--verbose'
                  , action='store_true'
                  , help='Control verbosity of output'
                  )

    p.add_argument( '-x'
                  , '--tix-percentage'
                  , default=1.0
                  , help='Percentage of EV0 ticket price to charge'
                  , type=float
                  )

    args = p.parse_args()
    plot_runs(args)
