#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os, re, sqlite3

def gen_nubase():
    # CODATA 2018
    me = 5.48579909065e-4 # electron mass in u
    keV2u = 1.07354410233e-6 # amount of u per 1 keV
    orb_e = {0: "bare", 1: "H-like", 2: "He-like", 3: "Li-like"} # ionic charge state related to its orbital electron count

    try:
        if os.path.exists("ionic_data.db"):
            os.remove("ionic_data.db")
        # create the database
        conn = sqlite3.connect('ionic_data.db')
        c = conn.cursor()
        # create the table for ionic_data 
        c.execute('''CREATE TABLE IONICDATA
                (A          INT         NOT NULL,
                ElEMENT     CHAR(2)     NOT NULL,
                Q           INT         NOT NULL,
                Z           INT         NOT NULL,
                N           INT         NOT NULL,
                ISOMERIC    CHAR(1)     NOT NULL,     
                TYPE        TEXT        NOT NULL,
                MASS        DOUBLE,
                MASSACC     REAL,
                SOURCE      TEXT,
                JPI         TEXT,
                HALFLIFE    TEXT,
                DECAYMODE   TEXT);''')
        ioniz_eng = pd.read_csv("./ionization.csv", comment='#')
        with open("./nubase2020.txt") as nubase:
            for _ in '_'*25:
                nubase.readline()
            for l in nubase:
                A, Z, Q, isomer_state, element, mass, jpi, mass_accuracy, decay_mode = int(l[:3]), int(l[4:7]), int(l[4:7]), l[7], re.split('(\d+)', l[11:16])[-1][:2], l[18:31].split(), ','.join(l[88:102].split()), l[31:42].split(), l[119:209]
                element = element[0] if element[1]==' ' else element
                stubs = l[69:80].split()
                half_life = stubs[0].rstrip('#') if len(stubs) > 0 else "n/a"
                half_life += ' ' + stubs[1] if (half_life[-1].isdigit() and len(stubs)>1) else ""
                if len(mass) == 0:
                    mass = " "
                    mass_accuracy = " "
                elif mass[0][-1] == "#":
                    mass, source = A + float(mass[0][:-1]) * keV2u, "estimated"
                    mass_accuracy = float(mass_accuracy[0][:-1]) * keV2u 
                else:
                    mass, source = A + float(mass[0]) * keV2u, "measured"
                    mass_accuracy = float(mass_accuracy[0]) * keV2u
                while True:
                    if mass == " ":
                        break
                    if Z == 0 and Q == 0:
                        pass
                    elif Q == Z:
                        mass = mass - Q*me + (14.4381*Q**2.39+1.55468e-6*Q**5.35)/1e3*keV2u # mass_Atom - n * me + Be(total)
                    elif Z-Q <= max(orb_e) and Q >= 0 and len(ioniz_eng[(ioniz_eng['Element']==element.split(" ")[0])&(ioniz_eng['Q']==Q)]) > 0:
                        mass += me - ioniz_eng[(ioniz_eng['Element']==element.split(" ")[0])&(ioniz_eng['Q']==Q)]["Ionization"].values[0]/1e3*keV2u
                        mass_accuracy = np.sqrt(mass_accuracy**2 + ioniz_eng[(ioniz_eng['Element']==element.split(" ")[0])&(ioniz_eng['Q']==Q)]["Uncertainty"].values[0]**2)
                        jpi = ''
                        decay_mode = ''
                    else:
                        break
                    c.execute("INSERT INTO IONICDATA (A,ELEMENT,Q,Z,N,ISOMERIC,TYPE,MASS,MASSACC,SOURCE,HALFLIFE, JPI, DECAYMODE)\
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (A, element, Q, Z, A-Z, isomer_state, orb_e[int(Z-Q)], mass, mass_accuracy, source, half_life, jpi, decay_mode))
                    Q -= 1
        conn.commit()
        conn.close()
    except FileNotFoundError:
        print("Error: cannot find the files of nubase2020 and ionization energy!")

if __name__ == '__main__':
    gen_nubase()
