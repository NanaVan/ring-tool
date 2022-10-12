#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import os, sqlite3
from gen_nubase import gen_nubase

class Utility(object):
    '''
    A class encapsulating some utility methods for the quick calculation during an isochronous Schottky beam time.
    It is a useful tool to calculate a specific ion (bare/H-like/He-like/Li-like).
    It is intended to be imported, e.g., into an IPython session, to be used for the ion identification.
    '''
    # CODATA 2018
    c = 299792458 # speed of light in m/s
    e = 1.602176634e-19 # elementary charge in C
    me = 5.48579909065e-4 # electron mass in u
    u2kg = 1.66053906660e-27 # amount of kg per 1 u
    MeV2u = 1.07354410233e-3 # amount of u per 1 MeV
    orb_e = {0: "bare", 1: "H-like", 2: "He-like", 3: "Li-like"} # ionic charge state related to its orbital electron count

    def __init__(self, cen_freq, span, L_CSRe=128.8, nubase_update=False, verbose=True):
        '''
        load the ionic data from disk, if any, otherwise build it and save it to disk
        the ionic data contain masses of nuclides in the bare, H-like, He-like, Li-like charge states.
        Besides, the half-lives of the corresponding atoms, and the isospin and charity of the corresponding bare ions are included as a reference.

        cen_freq:       center frequency of the spectrum [MHz]
        span:           span of the spectrum [kHz]
        L_CSRe:         circumference of CSRe [m], default value 128.8
        '''
        self.cen_freq = cen_freq # MHz
        self.span = span # kHz
        self.L_CSRe = L_CSRe # m
        self.verbose = verbose
        # check database ionic_data.db exists, if not create one
        if nubase_update or ((not nubase_update) and (not os.path.exists("./ionic_data.db"))):
            gen_nubase()
        self.conn = sqlite3.connect("./ionic_data.db")
        self.cur = self.conn.cursor()
        # check table ioncidata exists
        if self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' and name='IONICDATA'").fetchone()[0] == 0:
            gen_nubase()

    def set_cen_freq(self, cen_freq):
        '''
        set a new center frequency of the spectrum [MHz] 
        '''
        self.cen_freq = cen_freq # MHz

    def set_span(self, span):
        '''
        set a new span of the spectrun [kHz]
        '''
        self.span = span # kHz

    def set_L_CSRe(self, L_CSRe):
        '''
        set the adjusted circumference of CSRe [m]
        '''
        self.L_CSRe = L_CSRe

    def set_ion(self, ion, isometric_state=None):
        '''
        set the target ion, to be input in the format of AElementQ, e.g., 3He2
        '''
        element = ''.join(c for c in ion if c.isalpha())
        A, Q = map(int, ion.split(element))
        result = self.cur.execute("SELECT A, ELEMENT, Q, Z, ISOMERIC, TYPE, MASS, HALFLIFE FROM IONICDATA WHERE A=? AND ELEMENT=? AND Q=?", (A, element, Q)).fetchall()
        if len(result) == 0:
            print("Error: ion is too rare! please input another.")
        elif len(result) == 1:
            result = result[-1]
            if (isometric_state is None) or (isometric_state==result[4]):
                self.A, self.element, self.Q, self.Z, self.isometric_state, self.type, self.mass, self.half_life = result
            else:
                print("Warning: given ion with different isometric state {:} found. Please check and recall the function in the format of 'set_ion(AElementQ, isometric_state)'".format(result[4]))
        else:
            if isometric_state is None:
                print("Warning: given ion has several isometric states. Please choose one and recall the function in the format of 'set_ion(AElementQ, isometric_state)'")
                print("isometric_state: ", end='')
                for row in result:
                    print("{:} ".format(row[4]), end='')
                print("")
            else:
                result = self.cur.execute("SELECT A, ELEMENT, Q, Z, ISOMERIC, TYPE, MASS, HALFLIFE FROM IONICDATA WHERE A=? AND ELEMENT=? AND Q=? AND ISOMERIC=?", (A, element, Q, isometric_state)).fetchone()
                if len(result) == 0:
                    print("Error: ion is too rare! please input another.")
                else:
                    self.A, self.element, self.Q, self.Z, self.isometric_state, self.type, self.mass, self.half_life = result

    def set_gamma(self, gamma):
        '''
        set the Lorentz factor of the target ion
        '''
        self.gamma = gamma
        self.beta = np.sqrt(1 - self.gamma**-2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # magnetic rigidity in Tm
        self.energy = (self.gamma - 1) / self.MeV2u # kinetic energy in MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # revolution frequency in MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int) # selected harmonic numbers of the peaks viewed through the frequency window
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # peak locations in kHz after deduction of the center frequency
        if self.verbose: self.show()

    def set_beta(self, beta):
        '''
        set the velocity of the target ion in unit of speed of light
        '''
        self.beta = beta
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_Brho(self, Brho):
        '''
        set the magnetic rigidity of the target ion in Tm
        '''
        self.Brho = Brho # Tm
        gamma_beta = self.Brho / self.mass * self.Q / self.c / self.u2kg * self.e
        self.beta = gamma_beta / np.sqrt(1 + gamma_beta**2)
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_energy(self, energy):
        '''
        set the kinetic energy of the target ion in MeV/u
        '''
        self.energy = energy # MeV/u
        self.gamma = 1 + self.energy * self.MeV2u
        self.beta = np.sqrt(1 - self.gamma**-2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.rev_freq = self.beta * self.c / self.L_CSRe / 1e6 # MHz
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_rev_freq(self, rev_freq):
        '''
        set the revolution frequency of the target ion in MHz
        '''
        self.rev_freq = rev_freq # MHz
        self.beta = self.rev_freq * self.L_CSRe / self.c * 1e6
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def set_peak_loc(self, peak_loc, harmonic):
        '''
        set the peak location of the target ion in kHz after deduction of the center frequency
        in order to unambiguously deduce the revolution frequency, the harmonic number must also be specified
        '''
        self.rev_freq = (self.cen_freq + peak_loc/1e3) / harmonic # MHz
        self.beta = self.rev_freq * self.L_CSRe / self.c * 1e6
        self.gamma = 1 / np.sqrt(1 - self.beta**2)
        self.Brho = self.gamma * self.beta * self.mass / self.Q * self.c * self.u2kg / self.e # Tm
        self.energy = (self.gamma - 1) / self.MeV2u # MeV/u
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        self.harmonic = np.arange(np.ceil(lower_freq/self.rev_freq), np.floor(upper_freq/self.rev_freq)+1).astype(int)
        self.peak_loc = (self.harmonic * self.rev_freq - self.cen_freq) * 1e3  # kHz
        if self.verbose: self.show()

    def show(self):
        '''
        print all the kinematic and spectroscopic parameters of the target ion
        '''
        print('-' * 10)
        print("target ion\t\t{:d}{:s}{:d}".format(self.A,self.element,self.Q))
        print("charge state\t\t{:s}".format(self.type))
        print("isometric state\t\t{:s}".format(self.isometric_state))
        print("γ\t\t\t{:.6g}".format(self.gamma))
        print("β\t\t\t{:.6g}".format(self.beta))
        print("Bρ\t\t\t{:.6g} Tm".format(self.Brho))
        print("kinetic energy\t\t{:.6g} MeV/u".format(self.energy))
        print("ring circumference\t{:g} m".format(self.L_CSRe))
        print("revolution frequency\t{:.6g} MHz".format(self.rev_freq))
        print("center frequency\t{:g} MHz".format(self.cen_freq))
        print("span\t\t\t{:g} kHz".format(self.span))
        print("peak location(s)\t" + ', '.join(["{:.0f}".format(item) for item in self.peak_loc]) + " kHz")
        print("harmonic number(s)\t" + ', '.join(["{:d}".format(item) for item in self.harmonic]))
        print("atomic half-life\t{:s}".format(self.half_life))
   
    def help(self):
        '''
        display all the available functions of the class: Utility
        '''
        print('--' * 10 + '\n')
        print('Display all avaliable functions of the Utility\n')
        print("Input Only:")
        print("set_ion(ion, isometric_state=None)\n\tset the target ion, to be input in the format of AEmlementQ, e.g., 3H2")
        print("set_cen_freq(cen_freq)\n\tset a new center frequency of the spectrum [MHz]")
        print("set_span(span)\n\tset a new span of the spectrum [kHz]")
        print("set_L_CSRe(L_CSRe)\n\tset the adjusted circumference of CSRe [m]")
        print("Display the estimation result after input:")
        print("set_energy(energy)\n\tset the kinetic energy the target ion [MeV/u]")
        print("set_gamma(gamma)\n\tset the Lorentz factor of the target ion")
        print("set_beta(beta)\n\tset the velocity of the target ion in unit of speed of light")
        print("set_Brho(Brho)\n\tset the magnetic rigidity of the target ion [Tm]")
        print("set_rev_freq(rev_freq)\n\tset the revolution frequency of the target ion [MHz]")
        print("set_peak_loc(peak_loc, harmonic)\n\tset the peak location [kHz] of the target ion and corresponding harmonic for the calibration")
        print('\n' + '--' * 10) 

if __name__ == '__main__':
    utility = Utility(242.9, 500)
    utility.set_ion("58Ni27", 0)
    utility.set_energy(143.92)
