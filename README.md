`gen_nubase.py`: generate the database `ionic_data.db`, including 
                table `IONICDATA`:      the information of the ions
                                        A, ElEMENT, Q, Z, N, ISOMERIC, TYPE, MASS, MASSACC, SOURCE, JPI, HALFLIFE, DECAYMODE
                                        (A, element, Q, Z, N, isomeric_state, type of charge state, mass [u], mass_accuracy [keV], mass_source ('estimated'|'measured'), isospin_parity, half_life, decay_mode)

`plot_nubase.py`: give plots based on the database `ionic_data.db`.
                table `IONICDATA`:      generated from `gen_nubase`
                                        * immutable during the program.
                table `LIFEDATA`:       the information of the half-life (no isomer).
                                        ELEMENT, Z, N, HALFLIFE
                                        (element, Z, N, half_life)
                                        * temp. it will be deleted after plotting.
                table `MASSACCDATA`:    the information of the mass accuracy (no isomer).
                                        ELEMENT, Z, N, MASSACC, SOURCE
                                        (element, Z, N, mass_accuracy, mass_source)
                                        * temp. it will be deleted after plotting.
                table `MASSEXCDATA`:    the information of the mass excess (bare, no isomer).
                                        ELEMENT, Z, N, MASSEXC
                                        (element, Z, N, mass_excess)
                                        * temp. it will be deleted after plotting.

`utility.py`: a tool to show useful information of a targeted ion, based on the conversions between ion's velocity (β and γ), revolution frequency, magnetic rigidity, kinetic energy, as well as the peak location in the Schottky spectrum when the center frequency and span are given.
                table `IONICDATA`:      generated from `gen_nubase` 
                                        * immutable during the program.

`iid.py`: a tool to estimate the spectrum of ions given the LISE++ simulation result.
                table `IONICDATA`:      generated from `gen_nubase` 
                                        * immutable during the program.
                table `LPPDATA`:        the information of ions' yield from LISE++
                                        A, ELEMENT, Q, ION, YIELD
                                        (A, element, Q, AElementQ, ion_yield)
                                        * immutable during the program 
                table `OBSERVEDION`:    the information of observed ions from LISE++, depending on Schottky setting
                                        ID, A, ELEMENT, Q, Z, N, MASS, SOURCE, ION, YIELD, PEAKLOC, HARMONIC, REVFREQ, TYPE, ISOMERIC, HALFLIFE, WEIGHT 
                                        (sort of weight(desc), A, element, Q, Z, N, mass, mass_source, AElementQ, ion_yield, peak_loc, harmonic, rev_freq, type of charge state, isomertic_state, half_life, weight)
                                        * recreate after input parameters change
                table `TOTALION`:       the information of observed ions for GUI displaying
                                        ID, ION, TYPE, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE
                                        (sort of weight(desc), AElementQ, type of charge state, isomertic_state, ion_yield, weight, harmonic, peak_loc, rev_freq, half_life)
                table `BAREION`, `HLIKEION`, `HELIKEION`, `LILIKEION`:
                                        only exists if the certain type exists for GUI displaying
                                        ID, ION, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE
                                        (sort of weight(desc), AElementQ, isomertic_state, ion_yield, weight, harmonic, peak_loc, rev_freq, half_life)
                table `SEARCHION`:
                                        only exists if the search function works, containing only the valid ions
                                        ID, ION, ISOMERIC, YIELD, WEIGHT, HARMONIC, PEAKLOC, REVFREQ, HALFLIFE
                                        (sort of weight(desc), AElementQ, isomertic_state, ion_yield, weight, harmonic, peak_loc, rev_freq, half_life)
                
