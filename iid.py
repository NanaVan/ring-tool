#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import re, sqlite3

from utility import Utility
from plot_nubase import plot_heatmap

class IID(Utility):
    '''
    A script for auto calibrating the ion identification result based on the input Schottky spectrum
    '''
    def __init__(self, lppion, cen_freq, span, n_peak=10, GUI_mode=False, L_CSRe=128.8, nubase_update=False, verbose=False):
        '''
        extract all the secondary fragments and their respective yields calculated by LISE++
        (including Mass, Half-life, Yield of all the fragments)
        lppion:     LISE++ output file to be loaded
        cen_freq:   center frequency of the spectrum in MHz
        span:       span of the spectrum in kHz
        n_peak:     number of peaks to be identified
        L_CSRe:     circumference of CSRe in m, default value 128.8
        '''
        self.n_peak = n_peak
        self.sigma = 1.0e-6
        self.GUI_mode = GUI_mode
        super().__init__(cen_freq, span, L_CSRe, nubase_update, verbose)
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_name = [(part[0],) for part in self.cur.fetchall() if (part[0] != 'IONICDATA')]
        if len(table_name) > 0:
            self.cur.executescript(';'.join(["DROP TABLE IF EXISTS %s" %i for i in table_name]))
            self.conn.commit()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS LPPDATA
                (A          INT         NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Q           INT         NOT NULL,
                ION         TEXT        NOT NULL,
                YIELD       REAL);''')
        self.cur.execute("DELETE FROM LPPDATA")
        self.conn.commit()
        with open(lppion, encoding='latin-1') as lpp:
            while True:
                line = lpp.readline().strip()
                if line == "[D4_DipoleSettings]":
                    self.Brho = float(lpp.readline().strip().split()[2]) # Tm
                elif line == "[Calculations]":
                    break
            for line in lpp:
                segment = line.strip().split(',')[0].split()
                A, element, Q = re.split("([A-Z][a-z]?)", segment[0]+segment[1][:-1])
                self.cur.execute("INSERT INTO LPPDATA (A,ELEMENT,Q,ION,YIELD) VALUES (?,?,?,?,?)", (A, element, Q, ''.join([A,element,Q]), segment[-1][1:]))
            self.conn.commit()
        self.calc_peak()

    def prepare_result(self):
        self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='OBSERVEDION'")
        if self.cur.fetchone()[0] == 1:
            self.cur.execute("DROP TABLE OBSERVEDION")
            self.conn.commit()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS OBSERVEDION
                (ID         INT,
                A          INT          NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Q           INT         NOT NULL,
                Z           INT         NOT NULL,
                N           INT         NOT NULL,
                MASS        DOUBLE,
                SOURCE      TEXT,                
                ION         TEXT        NOT NULL,
                YIELD       REAL,
                PEAKLOC     REAL,
                HARMONIC    INT,
                REVFREQ     REAL,     
                TYPE        TEXT,
                ISOMERIC    CHAR(1),
                HALFLIFE    TEXT,
                WEIGHT      DOUBEL,
                PRIMARY KEY (ID));''')
        self.cur.execute("DELETE FROM OBSERVEDION")
        self.cur.execute("INSERT INTO OBSERVEDION(A,ELEMENT,Q,Z,N,MASS,SOURCE,ION,YIELD,TYPE,ISOMERIC,HALFLIFE) \
                SELECT LPPDATA.A, LPPDATA.ELEMENT, LPPDATA.Q, IONICDATA.Z, IONICDATA.N, IONICDATA.MASS, IONICDATA.SOURCE, LPPDATA.ION, LPPDATA.YIELD, IONICDATA.TYPE, IONICDATA.ISOMERIC, IONICDATA.HALFLIFE \
                FROM IONICDATA \
                INNER JOIN LPPDATA ON IONICDATA.Q=LPPDATA.Q AND IONICDATA.ELEMENT=LPPDATA.ELEMENT AND IONICDATA.A=LPPDATA.A")
        # reset the yield of the isometric_state
        result = self.cur.execute("SELECT YIELD, ION, ISOMERIC FROM OBSERVEDION WHERE ISOMERIC!=0").fetchall()
        # yield(isometric_state) = yield(bare) * 10**isometric_state
        re_set = [(item[0]*10**(-int(item[2])), item[1], item[2]) for item in result]
        self.cur.executemany("UPDATE OBSERVEDION SET YIELD=? WHERE ION=? AND ISOMERIC=?", re_set)
        self.conn.commit()

    def calc_peak(self):
        '''
        calculate peak locations of the Schottky signals from secondary fragments visible in the pre-defined frequency range
        '''
        self.prepare_result()
        # calculate rev_freq, weight
        self.cur.execute("SELECT MASS, Q, YIELD, ION, ISOMERIC FROM OBSERVEDION")
        mass, Q, ion_yield, ion, isometric_state = np.array(self.cur.fetchall()).T
        mass, Q, ion_yield = mass.astype(np.float64), Q.astype(np.float64), ion_yield.astype(np.float64)
        gamma_beta = self.Brho / mass * Q / self.c / self.u2kg * self.e
        beta = gamma_beta / np.sqrt(1 + gamma_beta**2)
        gamma = 1 / np.sqrt(1 - beta**2)
        energy = (gamma - 1) / self.MeV2u # MeV/u
        rev_freq = beta * self.c / self.L_CSRe / 1e6 # MHz
        weight = ion_yield * Q**2 * rev_freq**2
        lower_freq, upper_freq = self.cen_freq - self.span/2e3, self.cen_freq + self.span/2e3 # MHz, MHz
        i = 0
        # load the result into table observedion
        while (i < len(ion)):
            temp = self.cur.execute("SELECT * FROM OBSERVEDION WHERE ION=? AND ISOMERIC=?", (ion[i],isometric_state[i])).fetchone()
            self.cur.execute("DELETE FROM OBSERVEDION WHERE ION=? AND ISOMERIC=?", (ion[i],isometric_state[i]))
            harmonics = np.arange(np.ceil(lower_freq/rev_freq[i]), np.floor(upper_freq/rev_freq[i])+1).astype(int)
            # filter harmonics
            if len(harmonics) == 1:
                self.cur.execute("INSERT INTO OBSERVEDION(A,ELEMENT,Q,Z,N,MASS,SOURCE,ION,YIELD,PEAKLOC,HARMONIC,REVFREQ,TYPE,ISOMERIC,HALFLIFE,WEIGHT) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (*temp[1:10], (harmonics[-1]*rev_freq[i]-self.cen_freq)*1e3, int(harmonics[-1]), rev_freq[i], *temp[13:16], weight[i]))
            elif len(harmonics) > 1:
                re_set = [(*temp[1:10], (h*rev_freq[i]-self.cen_freq)*1e3, int(h), rev_freq[i], *temp[13:16], weight[i]) for h in harmonics]
                self.cur.executemany("INSERT INTO OBSERVEDION(A,ELEMENT,Q,Z,N,MASS,SOURCE,ION,YIELD,PEAKLOC,HARMONIC,REVFREQ,TYPE,ISOMERIC,HALFLIFE,WEIGHT) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", re_set)
            else:
                pass
            i += 1
        self.conn.commit()
        # set ID with sort of weight (descrease)
        temp = self.cur.execute("SELECT ION, ISOMERIC, HARMONIC FROM OBSERVEDION ORDER BY WEIGHT DESC").fetchall()
        re_set = [(i+1,*temp[i]) for i in range(len(temp))]
        self.cur.executemany("UPDATE OBSERVEDION SET ID=? WHERE ION=? AND ISOMERIC=? AND HARMONIC=?", re_set)
        self.conn.commit()
        if self.GUI_mode:
            self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_name = [(part[0],) for part in self.cur.fetchall() if (part[0] != 'IONICDATA' and part[0] != 'LPPDATA' and part[0] != 'OBSERVEDION')]
            if len(table_name) > 0:
                self.cur.executescript(';'.join(["DROP TABLE IF EXISTS %s" %i for i in table_name]))
                self.conn.commit()
            self.cur.execute('''CREATE TABLE TOTALION
                (ID         INT,
                ION         TEXT        NOT NULL,
                TYPE        TEXT,
                ISOMERIC    CHAR(1), 
                YIELD       REAL,
                WEIGHT      DOUBEL,
                HARMONIC    INT,         
                PEAKLOC     REAL,
                REVFREQ     REAL,     
                HALFLIFE    TEXT,
                PRIMARY KEY (ID));''')
            self.cur.execute("INSERT INTO TOTALION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION")
            type_name = self.cur.execute("SELECT TYPE FROM OBSERVEDION GROUP BY TYPE").fetchall()
            table_name = [''.join(item[0].split('-')).upper()+'ION' for item in type_name]
            for item in table_name:
                self.cur.execute('''CREATE TABLE %s
                (ID         INT,
                ION         TEXT        NOT NULL,
                ISOMERIC    CHAR(1), 
                YIELD       REAL,
                WEIGHT      DOUBEL,
                HARMONIC    INT,         
                PEAKLOC     REAL,
                REVFREQ     REAL,     
                HALFLIFE    TEXT,
                PRIMARY KEY (ID));''' % item)
            type_name = [(''.join(item[0].split('-')).upper()+'ION', item[0]) for item in type_name]
            for i, j in type_name:
                self.cur.execute("INSERT INTO %s SELECT ID, ION, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE TYPE=?" % i, (j,))
            self.conn.commit()
        else:
            self.show()

    def calc_gaussian_peak(self, table_name):
        '''
        table_name:     the table of ions to be calculate, e.g. OBSERVEDION, TOTALION, BAREION, HLIKEION, HELIKEION, LILIKEION, SEARCHION
        return the spectrum including all the selected ions in form of Gaussian peak
        default: sigma = delta f / rev_freq = 1e-5
        for each ion: width = sigma * harmonic * rev_freq / 1.66
        '''
        frequencyRange = np.linspace(-self.span/2, self.span/2, 8192) # kHz
        self.cur.execute("SELECT WEIGHT, HARMONIC, REVFREQ, PEAKLOC FROM %s" % table_name)
        weight, harmonic, rev_freq, peak_loc = np.array(self.cur.fetchall()).T
        N = len(weight)
        weight, harmonic, rev_freq, peak_loc = weight.reshape(N,1), harmonic.reshape(N,1), rev_freq.reshape(N,1), peak_loc.reshape(N,1)
        lim = weight.max() / 1e5
        freq_range = np.ones_like(weight).reshape(N,1) @ frequencyRange.reshape(1,8192)
        wid = (self.sigma * harmonic * rev_freq * 1e3 / 1.66)
        a = weight / (np.sqrt(2 * np.pi) * wid) * np.exp(-(freq_range - peak_loc)**2 / (2 * wid**2))
        a[a < lim] = lim
        ionPeaks = np.sum(a, axis=0)
        return frequencyRange, ionPeaks

    def search_ion(self, input_str):
        '''
        input_str:      the input string of ions to be selected, e.g. 3H1, 3H, H1, 3, H, 1+
        return the table of the ions containing the information of the input string
        '''
        if self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='SEARCHION'").fetchone()[0] == 1:
            self.cur.execute("DROP TABLE SEARCHION")
            self.conn.commit()
        self.cur.execute('''CREATE TABLE SEARCHION
            (ID         INT,
            ION         TEXT        NOT NULL,
            TYPE        TEXT,
            ISOMERIC    CHAR(1), 
            YIELD       REAL,
            WEIGHT      DOUBEL,
            HARMONIC    INT,         
            PEAKLOC     REAL,
            REVFREQ     REAL,     
            HALFLIFE    TEXT,
            PRIMARY KEY (ID));''')
        self.conn.commit()
        if input_str.isdigit() and self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE A=? OR Z=?", (int(input_str), int(input_str))).fetchone()[0] > 0: # match A or Z, e.g. 3
            self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE A=? OR Z=?", (int(input_str), int(input_str)))
            self.conn.commit()
            return
        elif  bool(re.fullmatch("[A-Z]", input_str)) or bool(re.fullmatch("[A-Z][a-z]", input_str)) and self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE ELEMENT=?", (input_str,)).fetchone()[0] > 0: # match element, e.g. H
            self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE ELEMENT=?", (input_str,))
            self.conn.commit()
            return
        elif bool(re.fullmatch("(\d+)[A-Za-z]+(\d+)", input_str)) and self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE ION=?", (input_str,)).fetchone()[0] > 0: # match AElementQ, e.g. 3H1
            self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE ION=?", (input_str,))
            self.conn.commit()
            return
        elif bool(re.fullmatch("(\d+)[A-Z][a-z]", input_str)) or bool(re.fullmatch("(\d+)[A-Z]", input_str)): # match AElement, e.g. 3H
            _, str_A, str_Element = re.split('(\d+)', input_str)
            if self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE A=? AND ELEMENT=?", (int(str_A), str_Element)).fetchone()[0] > 0:
                self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE A=? AND ELEMENT=?", (int(str_A), str_Element))
                self.conn.commit()
                return
        elif bool(re.fullmatch("[A-Z](\d+)", input_str)) or bool(re.fullmatch("[A-Z][a-z](\d+)", input_str)): # match ElementQ, e.g. H1
            str_Element, str_Q, _ = re.split('(\d+)', input_str)
            if self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE Q=? AND ELEMENT=?", (int(str_Q), str_Element)).fetchone()[0] > 0:
                self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE Q=? AND ELEMENT=?", (int(str_Q), str_Element))
                self.conn.commit()
                return
        elif (input_str[-1] == '+') and input_str[:-1].isdigit() and self.cur.execute("SELECT count(ID) FROM OBSERVEDION WHERE Q=?", (int(input_str[:-1]),)).fetchone()[0] > 0: # match charge, e.g. 1+
            self.cur.execute("INSERT INTO SEARCHION SELECT ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE FROM OBSERVEDION WHERE Q=?", (int(input_str[:-1]),))
            self.conn.commit()
            return
        else:
            self.cur.execute("DROP TABLE SEARCHION")
            self.conn.commit()
            return
        self.cur.execute("DROP TABLE SEARCHION")
        self.conn.commit()

    def set_ion(self, ion, isometric_state):
        '''
        override the function from the Utility.set_ion()
        set the target ion, to be input in the format of AElementQ, e.g., 3He2
        ion:        a string in the format of AElementQ, e.g., 3He2 
        isometric_state: a integer of isometric state, e.g., 0
        '''
        element = ''.join(c for c in ion if c.isalpha())
        A, Q = map(int, ion.split(element))
        result = self.cur.execute("SELECT A, ELEMENT, Q, Z, MASS, ION, YIELD, TYPE, ISOMERIC, HALFLIFE FROM OBSERVEDION WHERE ION=? AND ISOMERIC=?", (ion, isometric_state)).fetchone()
        if len(result) == 0:
            print("Error: ion given is not existed in the fragments!")
            return
        self.A, self.element, self.Q, self.Z, self.mass, self.ion, self.ion_yield, self.type, self.isometric_state, self.half_life = result

    def calibrate_Brho(self, Brho):
        '''
        using the measured Brho with the identified ion to calibrate
        Brho:       the magnetic rigidity of the target ion in Tm
        '''
        self.Brho = Brho
        return self.calc_peak()

    def calibrate_peak_loc(self, ion, isometric_state, peak_loc, harmonic):
        '''
        using the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        isometric_state: a integer of isometric state, e.g., 0
        peak_loc:   peak location in kHz after deduction of the center frequency
        harmonic:   harmonic number
        '''
        rev_freq = (self.cen_freq + peak_loc/1e3) / harmonic # MHz
        self.set_ion(ion, isometric_state)
        self.set_rev_freq(rev_freq)
        return self.calc_peak()
    
    def calibrate_rev_freq(self, ion, isometric_state, peak_loc_1, peak_loc_2):
        '''
        using the measured revolution frequency with the identified ion to calibrate the magnetic rigidity of CSRe
        ion:        a string in the format of AElementQ, e.g., 3He2 
        isometric_state: a integer of isometric state, e.g., 0
        peak_loc_1: peak location in kHz after deduction of the center frequency
        peak_loc_2: another peak location belonging to the same ions but differed by one harmonic number
        '''
        rev_freq = np.absolute(peak_loc_2-peak_loc_1) / 1e3 # MHz
        self.set_ion(ion, isometric_state)
        self.set_rev_freq(rev_freq)
        return self.calc_peak()

    def update_cen_freq(self, cen_freq):
        '''
        set a new center frequency of the spectrum in MHz
        '''
        self.cen_freq = cen_freq # MHz
        self.calc_peak()

    def update_span(self, span):
        '''
        set a new span of the spectrum in kHz
        '''
        self.span = span # kHz
        self.calc_peak()

    def update_n_peak(self, n_peak):
        '''
        set a new number of peaks to be shown in the output
        '''
        self.n_peak = n_peak
        self.show()

    def update_L_CSRe(self, L_CSRe):
        '''
        set the central orbital length of beams in CSRe in m
        '''
        self.L_CSRe = L_CSRe # m
        self.calc_peak()

    def plot_yield(self, z_min=None, z_max=None, n_min=None, n_max=None, annotated=False):
        '''
        plot chart of the nuclides diplaying the yield (ignoring isomers)
        '''
        #self.cur.execute("DROP TABLE TEMP")
        # create table temp
        self.cur.execute("CREATE TABLE TEMP (A INT, Q INT, TYPE TEXT, ELEMENT CHAR(2), Z INT, N INT, YIELD REAL);")
        self.conn.commit()
        self.cur.execute("INSERT INTO TEMP (A, Q, TYPE, ELEMENT, Z, N) SELECT A, Q, TYPE, ELEMENT, Z, N FROM IONICDATA WHERE ISOMERIC='0'")
        yield_min = self.cur.execute("SELECT min(YIELD) FROM LPPDATA").fetchone()[0]
        self.cur.execute("UPDATE TEMP SET YIELD=?", (yield_min*0.1, ))
        result = self.cur.execute("SELECT YIELD, A, ELEMENT, Q FROM LPPDATA").fetchall()
        self.cur.executemany("UPDATE TEMP SET YIELD=? WHERE A=? AND ELEMENT=? AND Q=?", result)
        self.conn.commit()
        charge_range = []
        for charge_type in ['bare', 'H-like', 'He-like', 'Li-like']:
            num = self.cur.execute("SELECT count(*) FROM TEMP WHERE TYPE=?", (charge_type,)).fetchone()[0]
            if self.cur.execute("SELECT sum(YIELD) FROM TEMP WHERE TYPE=?", (charge_type,)).fetchone()[0] <= yield_min*num:
                self.cur.execute("DELETE FROM TEMP WHERE TYPE=?", (charge_type,))
            else:
                charge_range.append(charge_type)
        # prepare the data for displaying
        Z_min, Z_max, N_min, N_max = self.cur.execute("SELECT min(Z), max(Z), min(N), max(N) FROM TEMP WHERE YIELD>=?", (yield_min,)).fetchone()
        z_min = Z_min if z_min == None or z_min < Z_min else z_min
        z_max = Z_max if z_max == None or z_max > Z_max else z_max
        n_min = N_min if n_min == None or n_min < N_min else n_min
        n_max = N_max if n_max == None or n_max > N_max else n_max
        z_range = np.arange(z_min, z_max+1)
        n_range = np.arange(n_min, n_max+1)
        data, data_annote = [], []
        for charge_type in charge_range:
            for Z in z_range:
                self.cur.execute("SELECT N, YIELD, ELEMENT FROM TEMP WHERE Z=? AND TYPE=? AND N>=? AND N<=? ORDER BY N", (int(Z), charge_type, N_min, N_max))
                result = np.array([[*row] for row in self.cur.fetchall()]).T
                temp_yield, temp_element = yield_min*0.01*np.ones_like(n_range).astype(np.float64), np.array(['  ' for i in n_range])
                if len(result) > 0: temp_yield[(result[0].astype(int)-N_min)] = result[1].astype(np.float64)
                if len(result) > 0: temp_element[(result[0].astype(int)-N_min)] = result[2]
                temp_yield.reshape(1,len(n_range))
                temp_element.reshape(1,len(n_range))
                if Z == z_range[0]:
                    data_temp = temp_yield
                    data_annote_temp = temp_element
                else:
                    data_temp = np.vstack((data_temp, temp_yield))
                    data_annote_temp = np.vstack((data_annote_temp, temp_element))
            data.append(data_temp)
            data_annote.append(data_annote_temp)
        self.cur.execute("DROP TABLE TEMP")
        self.conn.commit()
        pproj = plot_heatmap(np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5))
        pproj.heatmap(data, data_annote, charge_range, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':['{:}'.format(n) if n%5==0 else '' for n in n_range], 'y_ticklabels':['{:}'.format(z) if z%5==0 else '' for z in z_range], 'xlabel':'N', 'ylabel':'Z'}, cbar_ticklabels=[], cbar_label='Yield', cbar_kw=dict(extend='max'), annotated=annotated, cmap='Oranges')

    def plot_weight(self, z_min=None, z_max=None, n_min=None, n_max=None, annotated=False):
        '''
        plot chart of the nuclides diplaying the weight at the Schottky spectrum (ignoring isomers)
        '''
        #self.cur.execute("DROP TABLE TEMP")
        # create table temp
        self.cur.execute("CREATE TABLE TEMP (TYPE TEXT, ELEMENT CHAR(2), Z INT, N INT, WEIGHT REAL);")
        self.conn.commit()
        self.cur.execute("INSERT INTO TEMP (TYPE, ELEMENT, Z, N) SELECT TYPE, ELEMENT, Z, N FROM IONICDATA WHERE ISOMERIC='0'")
        weight_min = self.cur.execute("SELECT min(WEIGHT) FROM OBSERVEDION WHERE ISOMERIC='0'").fetchone()[0]
        self.cur.execute("UPDATE TEMP SET WEIGHT=?", (weight_min*0.1, ))
        result = self.cur.execute("SELECT WEIGHT, ELEMENT, N, TYPE FROM OBSERVEDION WHERE ISOMERIC='0'").fetchall()
        self.cur.executemany("UPDATE TEMP SET WEIGHT=? WHERE ELEMENT=? AND N=? AND TYPE=?", result)
        self.conn.commit()
        charge_range = []
        for charge_type in ['bare', 'H-like', 'He-like', 'Li-like']:
            num = self.cur.execute("SELECT count(*) FROM TEMP WHERE TYPE=?", (charge_type,)).fetchone()[0]
            if self.cur.execute("SELECT sum(WEIGHT) FROM TEMP WHERE TYPE=?", (charge_type,)).fetchone()[0] <= weight_min*num:
                self.cur.execute("DELETE FROM TEMP WHERE TYPE=?", (charge_type,))
            else:
                charge_range.append(charge_type)
        # prepare the data for displaying
        Z_min, Z_max, N_min, N_max = self.cur.execute("SELECT min(Z), max(Z), min(N), max(N) FROM TEMP WHERE WEIGHT>=?", (weight_min,)).fetchone()
        z_min = Z_min if z_min == None or z_min < Z_min else z_min
        z_max = Z_max if z_max == None or z_max > Z_max else z_max
        n_min = N_min if n_min == None or n_min < N_min else n_min
        n_max = N_max if n_max == None or n_max > N_max else n_max
        z_range = np.arange(z_min, z_max+1)
        n_range = np.arange(n_min, n_max+1)
        data, data_annote = [], []
        for charge_type in charge_range:
            for Z in z_range:
                self.cur.execute("SELECT N, WEIGHT, ELEMENT FROM TEMP WHERE Z=? AND TYPE=? AND N>=? AND N<=? ORDER BY N", (int(Z), charge_type, N_min, N_max))
                result = np.array([[*row] for row in self.cur.fetchall()]).T
                temp_weight, temp_element = weight_min*0.01*np.ones_like(n_range).astype(np.float64), np.array(['  ' for i in n_range])
                if len(result) > 0: temp_weight[(result[0].astype(int)-N_min)] = result[1].astype(np.float64)
                if len(result) > 0: temp_element[(result[0].astype(int)-N_min)] = result[2]
                temp_weight.reshape(1,len(n_range))
                temp_element.reshape(1,len(n_range))
                #print("Z:{:}, weight:{:}".format(Z,temp_weight[-1]))
                #print("Z:{:}, weight:{:}".format(Z,temp_element[-1]))
                if Z == z_range[0]:
                    data_temp = temp_weight
                    data_annote_temp = temp_element
                else:
                    data_temp = np.vstack((data_temp, temp_weight))
                    data_annote_temp = np.vstack((data_annote_temp, temp_element))
            data.append(data_temp)
            data_annote.append(data_annote_temp)
        self.cur.execute("DROP TABLE TEMP")
        self.conn.commit()
        pproj = plot_heatmap(np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5))
        pproj.heatmap(data, data_annote, charge_range, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':['{:}'.format(n) if n%5==0 else '' for n in n_range], 'y_ticklabels':['{:}'.format(z) if z%5==0 else '' for z in z_range], 'xlabel':'N', 'ylabel':'Z'}, cbar_ticklabels=[], cbar_label='Weight', cbar_kw=dict(extend='max'), annotated=annotated, cmap='Oranges')

    def show(self):
        '''
        list the most prominent peaks in a Schottky spectrum sorted in a descending order
        '''
        print('-' * 16)
        print("center frequency\t{:g} MHz".format(self.cen_freq))
        print("span\t\t\t{:g} kHz".format(self.span))
        print("orbital length\t\t{:g} m".format(self.L_CSRe))
        print("Bρ\t\t\t{:.6g} Tm\n".format(self.Brho))
        self.cur.execute("SELECT WEIGHT, ION, HALFLIFE, YIELD, REVFREQ, PEAKLOC, HARMONIC FROM OBSERVEDION ORDER BY WEIGHT DESC")
        result = self.cur.fetchall()[:self.n_peak]
        print("Weight\t\tIon     Half-Life       Yield           RevFreq         PeakLoc Harmonic")
        for row in result:
            print("{:<8.2e}\t{:<7s}\t{:<11s}\t{:<9.2e}\t{:<8.6f}\t{:<4.0f}\t{:<3d}".format(*row[:-1], int(row[-1])))

    def help(self):
        '''
        override the function from Utility.help() 
        display all the available functions of the class: IID
        '''
        print('--' * 10 + '\n')
        print('Display all avaliable functions of the IID\n')
        print("calibrate_Brho(Brho)\n\tusing the measured Brho with the identified ion to calibrate\n\tBrho: the magnetic rigidity of the target ion [Tm]")
        print("calibrate_peak_loc(ion, peak_loc, harmonic)\n\tusing the measured peak location with the identified ion to calibrate the magnetic rigidity of CSRe\n\tion:\t\ta string in the format of AElementQ, e.g., 3H2\n\tpeak_loc:\tpeak location after deduction of the center frequency [kHz]\n\tharmonic:\tharmonic number")
        print("calibrate_rev_freq(ion, peak_loc_1, peak_loc_2)\n\tion:\t\ta string in the format of AElementQ, e.g., 3H2\n\tpeak_loc_1:\tpeak location after deduction of the center frequency [kHz]\n\tpeak_loc_2:\tanother peak location belonging to the same ions but differed by one harmonic number [kHz]")
        print("update_cen_freq(cen_freq)\n\tset a new center frequency of the spectrum [MHz]")
        print("update_span(span)\n\tset a new span of the spectrum [kHz]")
        print("update_n_peak(n_peak)\n\tset a new numer of peaks to be shown in the output")
        print("update_L_CSRe(L_CSRe)\n\tset the adjusted circumference of CSRe [m]")
        print('\n' + '--' * 10) 


if __name__ == "__main__":
    iid = IID("./86Kr36.lpp", 242.9, 1000)
    iid.calibrate_Brho(7.210)
