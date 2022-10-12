#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import matplotlib.lines as lines
import matplotlib.cbook as cbook
import sqlite3, os, re

class MidpointNormalize(colors.LogNorm):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [np.log(self.vmin), np.log(self.vcenter), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))

def decayModeSelect(text):
    if text[:1] == "A":
        return "A"
    elif text[:2] == "B-" or text[:3] == "2B-":
        return "B-"
    elif text[:2] == "B+" or text[:3] == "2B+":
        return "B+"
    elif text[:2] == "EC":
        return "EC"
    elif text[:2] == "SF":
        return "SF"
    elif text[:1] == "n" or text[:2] == "2n":
        return "n"
    elif text[:1] == "p" or text[:2] == "2p" or text[:2] == "3p":
        return "p"
    elif text[:2] == "IS":
        return "IS"
    else:
        return None

class plot_heatmap(object):
    '''
    clas for ploting the heatmap
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def heatmap(self, data, data_annote, subplot_titles=[], xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':[], 'y_ticklabels':[], 'xlabel':'', 'ylabel':''}, cbar_ticklabels=[], cbar_label='', cbar_kw={}, annotated=False, chinese=False, **kwargs):

        if chinese: self._chinese_typing()
        norm = MidpointNormalize(vmin=np.min(data), vcenter=np.min(data)*10, vmax=np.max(data))
        if len(subplot_titles) <= 0 or len(subplot_titles) > 4:
            raise ValueError("Valid charge state input!")
        elif len(subplot_titles) == 1:
            fig, ax = plt.subplots()
            im = self.heatmap_colorbar(data[0], ax=ax, xy_ticks=xy_ticks, cbar_ticklabels=cbar_ticklabels, cbar_label=cbar_label, cbar_kw=cbar_kw, norm=norm, **kwargs)
            if annotated:
                self.annotate_heatmap(im, data=data[0], data_annote=data_annote[0], valfmt="{x:2s}", threshold=np.min(data)*10)
            plt.show()     
            return
        elif len(subplot_titles) == 2:
            fig, axes = plt.subplots(nrows=1, ncols=2)
        elif len(subplot_titles) == 3:
            fig, axes = plt.subplots(nrows=3, ncols=1)
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2)
        for dat, ax, charge_state, dat_annote in zip(data, axes.flat, subplot_titles, data_annote):
            # plot the heatmap
            im = ax.pcolormesh(self.x, self.y, dat, norm=norm, **kwargs)
            # display the ticks...
            ax.set_xticks(xy_ticks['x_ticks']) if len(xy_ticks['x_ticks']) > 0 else ax.set_xticks(np.arange(dat.shape[1])+self.x[0]+0.5)
            ax.set_yticks(xy_ticks['y_ticks']) if len(xy_ticks['y_ticks']) > 0 else ax.set_yticks(np.arange(dat.shape[0])+self.y[0]+0.5)
            ax.set_xticklabels(xy_ticks['x_ticklabels']) if len(xy_ticks['x_ticklabels']) > 0 else ax.set_xticklabels(self.x)
            ax.set_yticklabels(xy_ticks['y_ticklabels']) if len(xy_ticks['y_ticklabels']) > 0 else ax.set_yticklabels(self.y)
            ax.set_xlabel(xy_ticks['xlabel']) 
            ax.set_ylabel(xy_ticks['ylabel'])
            ax.axis('equal')
            ax.set_title(charge_state)
        # element annotated
            if annotated:
                self.annotate_heatmap(im, data=dat, data_annote=dat_annote, valfmt="{x:2s}", threshold=np.min(data)*10)

        # create colorbar
        cbar = fig.colorbar(im, ax=axes.flatten(), **cbar_kw)
        cbar.set_label(cbar_label, rotation=-90, va="bottom")
        if len(cbar_ticklabels) != 0: cbar.set_ticklabels(cbar_ticklabels)
        plt.show()     
    
    def heatmap_legend(self, data, ax=None, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':[], 'y_ticklabels':[], 'xlabel':'', 'ylabel':''}, patch_list=[], legend_ticklabels=[], legend_title='', legend_kw={}, chinese=False, **kwargs):
        '''
        create a heatmap with patch legend

        data:               a 2D numpy array of shape (N, M)
        ax:                 A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
                            If not provided, use current axes or create a new one. Optional.
        xy_ticks:           A dictionary with arguments for the x, y ticks
                            `x_tick`:       tick for x
                            `y_tick`:       tick for y
                            `x_ticklabels`: tick labels for x
                            `y_ticklabels`: tick labels for y
                            `xlabel`:       label for x-axis
                            `ylabel`:       label for y-axis
        patch_list:         patch color range for legend ticks
        legend_ticklabels:  tick labels for legend
        legend_title:       title for legend
        legend_kw:          all other arguments for the legend. Optional
        **kwargs:           all other arguments are forwarded to `pcolormesh`        
        '''
        if chinese: self._chinese_typing()
        if not ax:
            ax = plt.gca()
        # plot the heatmap
        im = ax.pcolormesh(self.x, self.y, data, **kwargs)
        # create legend
        ax.legend(patch_list, legend_ticklabels, title=legend_title, **legend_kw)
        # display the ticks...
        ax.set_xticks(xy_ticks['x_ticks']) if len(xy_ticks['x_ticks']) > 0 else ax.set_xticks(np.arange(data.shape[1])+self.x[0]+0.5)
        ax.set_yticks(xy_ticks['y_ticks']) if len(xy_ticks['y_ticks']) > 0 else ax.set_yticks(np.arange(data.shape[0])+self.y[0]+0.5)
        ax.set_xticklabels(xy_ticks['x_ticklabels']) if len(xy_ticks['x_ticklabels']) > 0 else ax.set_xticklabels(self.x)
        ax.set_yticklabels(xy_ticks['y_ticklabels']) if len(xy_ticks['y_ticklabels']) > 0 else ax.set_yticklabels(self.y)
        ax.axis('equal')
        ax.set_xlabel(xy_ticks['xlabel']) 
        ax.set_ylabel(xy_ticks['ylabel']) 

        return im

    def heatmap_colorbar(self, data, ax=None, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':[], 'y_ticklabels':[], 'xlabel':'', 'ylabel':''}, cbar_ticklabels=[], cbar_label='', cbar_kw={}, chinese=False, **kwargs):
        '''
        create a heatmap with colorbar

        data:               a 2D numpy array of shape (N, M)
        ax:                 A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
                            If not provided, use current axes or create a new one. Optional.
        xy_ticks:           A dictionary with arguments for the x, y ticks
                            `x_tick`:       tick for x
                            `y_tick`:       tick for y
                            `x_ticklabels`: tick labels for x
                            `y_ticklabels`: tick labels for y
                            `xlabel`:       label for x-axis
                            `ylabel`:       label for y-axis
        cbar_ticklabels:     tick labels for the colorbar.
        cbar_label:          title for the colorbar.
        cbar_kw:            all other arguments for the colorbar. Optional.
        **kwargs:   all other arguments are forwarded to `pcolormesh`        
        '''
        if chinese: self._chinese_typing()
        if not ax:
            ax = plt.gca()
        # plot the heatmap
        im = ax.pcolormesh(self.x, self.y, data, **kwargs)
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
        if len(cbar_ticklabels) != 0: cbar.ax.set_yticklabels(cbar_ticklabels)
        # display the ticks...
        ax.set_xticks(xy_ticks['x_ticks']) if len(xy_ticks['x_ticks']) > 0 else ax.set_xticks(np.arange(data.shape[1])+self.x[0]+0.5)
        ax.set_yticks(xy_ticks['y_ticks']) if len(xy_ticks['y_ticks']) > 0 else ax.set_yticks(np.arange(data.shape[0])+self.y[0]+0.5)
        ax.set_xticklabels(xy_ticks['x_ticklabels']) if len(xy_ticks['x_ticklabels']) > 0 else ax.set_xticklabels(self.x)
        ax.set_yticklabels(xy_ticks['y_ticklabels']) if len(xy_ticks['y_ticklabels']) > 0 else ax.set_yticklabels(self.y)
        ax.axis('equal')
        ax.set_xlabel(xy_ticks['xlabel']) 
        ax.set_ylabel(xy_ticks['ylabel'])  

        return im

    def annotate_heatmap(self, im, data=None, data_annote=None, valfmt="", textcolors=("black", "white"), threshold=None, chinese=False, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
        if chinese: self._chinese_typing()

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = mpl.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j+self.x[0]+0.5, i+self.y[0]+0.5, valfmt(data_annote[i, j], None), clip_on=True, **kw)
                #text = ax.text(j+self.x[0]+0.5, i+self.y[0]+0.5, valfmt(data_annote[i, j], None), **kw)

    def heatmap_hatch(self, data, bounds, ax=None, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':[], 'y_ticklabels':[], 'xlabel':'', 'ylabel':''}, hatch_list=[], hatch_ticklabels=[], legend_title='', legend_kw={}, cbar_bin={'cbar_max': 130, 'cbar_range': []}, chinese=False, **kwargs):
        '''
        plot binary heatmap with patterns
        data:               a 2D numpy array of shape (N, M)
        ax:                 A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
                            If not provided, use current axes or create a new one. Optional.
        xy_ticks:           A dictionary with arguments for the x, y ticks
                            `x_tick`:       tick for x
                            `y_tick`:       tick for y
                            `x_ticklabels`: tick labels for x
                            `y_ticklabels`: tick labels for y
                            `xlabel`:       label for x-axis
                            `ylabel`:       label for y-axis
        bounds:             value of bound range, len: len(hatch_ticklabels)+1
                            [value_0, value_1, ... , value_n]
        hatch_list:         range of patterns, {'o', 'x', '*', '/', '+', '-', '.'} selected
        hatch_ticklabels:   tick labels for hatch patterns
        cbar_bin:           `cbar_max`: max level of color.greys
                            `cbar_range`: level value of color.greys for each hatch
        '''
        if chinese: self._chinese_typing()
        
        if not ax:
            ax = plt.gca()

        color_gray = mpl.cm.get_cmap('Greys', cbar_bin['cbar_max'])
        if len(cbar_bin['cbar_range']) == 0:
            cbar_bin['cbar_range'] = np.linspace(0, cbar_bin['cbar_max'], num=len(hatch_ticklabels), endpoint=True)
        if len(hatch_list) == 0:
            patterns = ['o', 'x', '*', '/', '+', '-', '.']
            hatch_list = [patterns[np.random.randint(0,7)]*np.random.randint(1,6) for i in range(len(hatch_ticklabels))]
        legend_handles = []
        for i, _hatch, color_num in zip(range(len(hatch_ticklabels)), hatch_list, cbar_bin['cbar_range']):
            ax.pcolor(self.x, self.y, np.ma.masked_outside(data, bounds[i], bounds[i+1]), hatch=_hatch, cmap=colors.ListedColormap([color_gray(color_num), color_gray(0)]))
            legend_handles.append(mpatches.Patch(facecolor=color_gray(color_num), hatch=_hatch, label=hatch_ticklabels[i]))
        ax.legend(handles=legend_handles, title=legend_title, **legend_kw)
        # display the ticks...
        ax.set_xticks(xy_ticks['x_ticks']) if len(xy_ticks['x_ticks']) > 0 else ax.set_xticks(np.arange(data.shape[1])+self.x[0]+0.5)
        ax.set_yticks(xy_ticks['y_ticks']) if len(xy_ticks['y_ticks']) > 0 else ax.set_yticks(np.arange(data.shape[0])+self.y[0]+0.5)
        ax.set_xticklabels(xy_ticks['x_ticklabels']) if len(xy_ticks['x_ticklabels']) > 0 else ax.set_xticklabels(self.x)
        ax.set_yticklabels(xy_ticks['y_ticklabels']) if len(xy_ticks['y_ticklabels']) > 0 else ax.set_yticklabels(self.y)
        ax.axis('equal')
        ax.set_xlabel(xy_ticks['xlabel']) 
        ax.set_ylabel(xy_ticks['ylabel'])

                

    def _chinese_typing(self):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False


class plot_nubase(plot_heatmap):
    '''
    class for displaying chart of the nuclides
    using the data from nubase2016
    '''

    def __init__(self, nubase_update=False):
        # check database ionic_data.db exists, if not create one
        if nubase_update or ((not nubase_update) and (not os.path.exists("./ionic_data.db"))):
            gen_nubase()
        self.conn = sqlite3.connect("./ionic_data.db")
        self.cur = self.conn.cursor()
        # check table ionicdata exists
        if self.cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' and name='IONICDATA'").fetchone()[0] == 0:
            gen_nubase()

        self.Z_range = np.arange(119)
        self.N_range = np.arange(181)
        super().__init__(np.arange(self.N_range[0]-0.5,self.N_range[-1]+1.5), np.arange(self.Z_range[0]-0.5, self.Z_range[-1]+1.5))

    def plot_life(self, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True):
        '''
        plot chart of the nuclides displaying half-lives
        '''
        time_unit = {'Yy': 31536000*1e24, 'Zy': 31536000*1e21, 'Ey': 31536000*1e18, 'Py': 31536000*1e15, 'Ty': 31536000*1e12, 'Gy': 31536000*1e9, 'My': 31536000*1e6, 'ky': 31536000*1e3, 'y': 31536000, 'd': 86400, 'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18, 'zs': 1e-21, 'ys': 1e-24}
        # create table lifedata 
        self.cur.execute("CREATE TABLE LIFEDATA (ELEMENT CHAR(2), Z INT, N INT, LIFE DOUBLE);")
        self.conn.commit()
        result = self.cur.execute("SELECT A, ELEMENT, Z, N, HALFLIFE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = []
        for row in result:
            temp = row[-1].split()
            if len(temp) == 1:
                if temp[0] == 'stbl':
                    re_set.append((*row[1:-1], 1e64))
                else:
                    re_set.append((*row[1:-1], -1))
            else:
                if temp[-1].isalpha():
                    try:
                        temp_0 = float(temp[0]) * time_unit[temp[-1]]
                    except:
                        temp_0 = float(''.join(re.split("(\d+)", temp[0])[1:])) * time_unit[temp[-1]]
                    re_set.append((*row[1:-1], temp_0))
                else:
                    re_set.append((*row[1:-1], -1))
        self.cur.executemany("INSERT INTO LIFEDATA (ELEMENT, Z, N, LIFE) VALUES (?,?,?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] if z_min == None or z_min < self.Z_range[0] else z_min
        z_max = self.Z_range[-1] if z_max == None or z_max > self.Z_range[-1] else z_max
        n_min = self.N_range[0] if n_min == None or n_min < self.N_range[0] else n_min
        n_max = self.N_range[-1] if n_max == None or n_max > self.N_range[-1] else n_max
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        # prepare the data for displaying
        for Z in Z_range:  
            self.cur.execute("SELECT N,LIFE,ELEMENT FROM LIFEDATA WHERE Z=? AND N>=? AND N<=? ORDER BY N", (int(Z), int(n_min), int(n_max)))
            result = np.array([[*row] for row in self.cur.fetchall()]).T
            temp_life, temp_element = -2*np.ones_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
            if len(result)>0: temp_life[result[0].astype(int)] = result[1].astype(np.float64)
            if len(result)>0: temp_element[result[0].astype(int)] = result[2]
            temp_life.reshape(1,len(N_range))
            temp_element.reshape(1,len(N_range))
            if Z == Z_range[0]:
                data = temp_life
                data_annote = temp_element
            else:
                data = np.vstack((data, temp_life))
                data_annote = np.vstack((data_annote, temp_element))
        data = np.ma.masked_where(data<=-2., data)
        data_annote = np.ma.masked_where(data_annote=='  ', data_annote)
        # drop table lifedata
        self.cur.execute("DROP TABLE LIFEDATA")
        self.conn.commit()
        # setup the plot
        time_limits = [-1.5, -0.5, 0.1, 3, 2*time_unit['m'], 1*time_unit['h'], 1*time_unit['d'], 1*time_unit['y'], 1*time_unit['Gy'], 1*time_unit['Ty']]
        time_limits_ticklabels = ['Unknown half-life', 'T < 0.1 s', '0.1 s '+'$\leq$'+ ' T < 3 s', '3 s '+'$\leq$'+' T < 2 m', '2 m ' +'$\leq$' +' T < 1 h', '1 h ''$\leq$'+' T < 1 d', '1 d '+'$\leq$'+' T < 1 y', '1 y '+'$\leq$'+' T < 1 Gy', 'T '+'$\geq$' + ' 1 Gy']
        bounds = np.array(time_limits)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.cm.get_cmap('magma', len(time_limits_ticklabels))
        patch_list = [mpatches.Patch(color=cmap(b)) for b in range(len(bounds))[:-1]]
        fig, ax = plt.subplots()
        im = self.heatmap_legend(data, ax=ax, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in N_range], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in Z_range], 'xlabel':'N', 'ylabel':'Z'}, patch_list=patch_list, legend_ticklabels=time_limits_ticklabels[::-1], legend_title='HALF LIFE', legend_kw=dict(loc='lower right'), norm=norm, cmap='magma_r')
        if add_sheline: self.add_shellline(ax)
        plt.show()

    def plot_mass_accuracy(self, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True):
        '''
        plot chart of the nuclides displaying the accuracy 'u' of masses [keV]
        '''
        keV2u = 1.07354410233e-6 # amount of u per 1 keV
        # create table massaccdata
        self.cur.execute("CREATE TABLE MASSACCDATA (ELEMENT CHAR(2), Z INT, N INT, MASSACC REAL, SOURCE TEXT);")
        self.conn.commit()
        max_mass = self.cur.execute("SELECT max(MASSACC) FROM IONICDATA WHERE SOURCE='measured'").fetchone()[0]
        result = self.cur.execute("SELECT ELEMENT, Z, N, MASSACC, SOURCE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = [row if row[-1] == "measured" else (*row[:3], max_mass*10, row[-1]) for row in result]
        self.cur.executemany("INSERT INTO MASSACCDATA (ELEMENT, Z, N, MASSACC, SOURCE) VALUES (?,?,?,?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] if z_min == None or z_min < self.Z_range[0] else z_min
        z_max = self.Z_range[-1] if z_max == None or z_max > self.Z_range[-1] else z_max
        n_min = self.N_range[0] if n_min == None or n_min < self.N_range[0] else n_min
        n_max = self.N_range[-1] if n_max == None or n_max > self.N_range[-1] else n_max
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        # prepare the data for displaying
        for Z in Z_range:  
            self.cur.execute("SELECT N,MASSACC,ELEMENT FROM MASSACCDATA WHERE Z=? AND N>=? AND N<=? ORDER BY N", (int(Z), int(n_min), int(n_max)))
            result = np.array([[*row] for row in self.cur.fetchall()]).T
            temp_massacc, temp_element = max_mass*20*np.ones_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
            if len(result)>0: temp_massacc[result[0].astype(int)] = result[1].astype(np.float64)
            if len(result)>0: temp_element[result[0].astype(int)] = result[-1]
            temp_massacc.reshape(1,len(N_range))
            temp_element.reshape(1,len(N_range))
            if Z == Z_range[0]:
                data = temp_massacc
                data_annote = temp_element
            else:
                data = np.vstack((data, temp_massacc))
                data_annote = np.vstack((data_annote, temp_element))
        data = np.ma.masked_where(data>max_mass*10, data)
        data_annote = np.ma.masked_where(data_annote=='  ', data_annote)
        # drop table massaccdata
        self.cur.execute("DROP TABLE MASSACCDATA")
        self.conn.commit()
        # setup the plot
        massacc_limits = [-0.1, keV2u, 10*keV2u, 100*keV2u, 1000*keV2u, max_mass, max_mass*20]
        massacc_limits_ticklabels = ['< 1 keV', '1 keV' + '$\leq$' + ' Uncertainty  < 10 keV', '10 keV ' + '$\leq$' + ' Uncertainty < 100 keV', '100 keV ' + '$\leq$' + ' Uncertainty < 1 MeV', '$\geq$' + '1 MeV', 'Unknown Mass']
        bounds = np.array(massacc_limits)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.cm.get_cmap('magma', len(massacc_limits_ticklabels))
        patch_list = [mpatches.Patch(color=cmap(b)) for b in range(len(bounds))[:-1]]
        fig, ax = plt.subplots()
        im = self.heatmap_legend(data, ax=ax, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in N_range], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in Z_range], 'xlabel':'N', 'ylabel':'Z'}, patch_list=patch_list, legend_ticklabels=massacc_limits_ticklabels, legend_title='MASS ACCURACY (keV)', legend_kw=dict(loc='lower right'), norm=norm, cmap='magma')
        if add_sheline: self.add_shellline(ax)
        plt.show()

    def plot_yield(self, include_estimated=True, add_sheline=True):
        '''
        plot chart of the nuclides displaying the yield
        * YIELDDATA needed, after iid.py

        include_estimated:  True for including the nuclides with the estimated mass excess. False for not including.
        '''
        #self.cur.execute("DROP TABLE YIELDDATA")
        # create table yieldcdata
        self.cur.execute("CREATE TABLE YIELDDATA (ELEMENT CHAR(2), Z INT, N INT, TYPE TEXT, HAVEISOMERIC INT, YIELD DOUBLE);")
        self.conn.commit()
        self.cur.execute("INSERT INTO YIELDDATA(ELEMENT,Z,N,YIELD,TYPE) \
                SELECT LPPDATA.ELEMENT, IONICDATA.Z, IONICDATA.N, LPPDATA.YIELD, IONICDATA.TYPE \
                FROM IONICDATA \
                INNER JOIN LPPDATA ON IONICDATA.Q=LPPDATA.Q AND IONICDATA.ELEMENT=LPPDATA.ELEMENT AND IONICDATA.A=LPPDATA.A")
        # update the whether have isomeric state 
        result = self.cur.execute("SELECT DISTINCT Z, N FROM YIELDDATA").fetchall()
        result = [self.cur.execute("SELECT Z, N, count(*) FROM IONICDATA WHERE Z=? AND N=? AND TYPE='bare'", item).fetchone() for item in result]
        re_set = [(0, item[0], item[1]) if item[-1] == 1 else (1, item[0], item[1]) for item in result]
        self.cur.executemany("UPDATE YIELDDATA SET HAVEISOMERIC=? WHERE Z=? AND N=?", re_set)
        self.conn.commit()
        z_min, z_max, n_min, n_max = self.cur.execute("SELECT min(Z), max(Z), min(N), max(N) FROM YIELDDATA").fetchone()
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        # check the number of nuclei type
        nuclei_type = self.cur.execute("SELECT DISTINCT TYPE FROM YIELDDATA").fetchall()
        nuclei_type = [_type[0] for _type in nuclei_type]
        # prepare the data for displaying
        data, data_annote = [], []
        for _type in nuclei_type:
            for Z in Z_range:  
                self.cur.execute("SELECT N,YIELD,ELEMENT FROM YIELDDATA WHERE Z=? AND N>=? AND N<=? AND TYPE=? ORDER BY N", (int(Z), int(n_min), int(n_max), _type))
                result = np.array([[*row] for row in self.cur.fetchall()]).T
                temp_yield, temp_element = np.zeros_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
                if len(result)>0: temp_yield[result[0].astype(int)-n_min] = result[1].astype(np.float64)
                if len(result)>0: temp_element[result[0].astype(int)-n_min] = result[2]
                temp_yield.reshape(1,len(N_range))
                temp_element.reshape(1,len(N_range))
                if Z == Z_range[0]:
                    data_temp = temp_yield
                    data_annote_temp = temp_element
                else:
                    data_temp = np.vstack((data_temp, temp_yield))
                    data_annote_temp = np.vstack((data_annote_temp, temp_element))
            data.append(data_temp)
            data_annote.append(data_annote_temp)
        # prepare isomer line
        data_isomer = self.cur.execute("SELECT N,Z FROM YIELDDATA WHERE HAVEISOMERIC=1 AND TYPE='bare'").fetchall()
        isomer_line = [([ion[0]+0.5, ion[0]+0.5], [ion[1]-0.5,ion[1]+0.5]) for ion in data_isomer] + [([ion[0]-0.5, ion[0]-0.5], [ion[1]-0.5,ion[1]+0.5]) for ion in data_isomer] + [([ion[0]-0.5, ion[0]+0.5], [ion[1]-0.5,ion[1]-0.5]) for ion in data_isomer] + [([ion[0]-0.5, ion[0]+0.5], [ion[1]+0.5,ion[1]+0.5]) for ion in data_isomer]
        self.cur.execute("DROP TABLE YIELDDATA")
        # plot heatmap
        norm = colors.LogNorm(vmin=min([x[x>0].min() for x in data]),vmax=max([x.max() for x in data]))
        axs = self.heatmap(data, data_annote, subplot_titles=nuclei_type, norm=norm, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':[], 'y_ticklabels':[],'xlabel': 'N', 'ylabel': 'Z'}, cbar_label='yield', cbar_kw={'location':'bottom'})
        # add nuclear chart line and isomer annotated line
        for ax in axs:
            self.plot_blank_chart(include_estimated=True, z_min=z_min, z_max=z_max, n_min=n_min, n_max=n_max, add_sheline=add_sheline, embedded=True, ax=ax)
            for x, y in isomer_line:
                line = lines.Line2D(x, y, color='darkorange', axes=ax)
                ax.add_line(line)
            ax.set_xlim([n_min-0.5, n_max+0.5])
            ax.set_ylim([z_min-0.5, z_max+0.5])
        plt.show()

    def plot_mass_excess(self, include_estimated=True, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True):
        '''
        plot chart of the nuclides displaying the mass excess [keV] 

        include_estimated:  True for including the nuclides with the estimated mass excess. False for not including.
        '''
        self.cur.execute("DROP TABLE MASSEXCDATA")
        # create table massaxcdata
        self.cur.execute("CREATE TABLE MASSEXCDATA (ELEMENT CHAR(2), Z INT, N INT, MASSEXC DOUBLE);")
        self.conn.commit()
        result = self.cur.execute("SELECT ELEMENT, Z, N, A, MASS, SOURCE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = [(*row[:3], (row[4]-row[3])*1e6) for row in result] if include_estimated else [(*row[:3], (row[4]-row[3])*1e6) if row[-1] == 'measured' else (*row[:3], 0) for row in result]
        self.cur.executemany("INSERT INTO MASSEXCDATA (ELEMENT, Z, N, MASSEXC) VALUES (?,?,?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] if z_min == None or z_min < self.Z_range[0] else z_min
        z_max = self.Z_range[-1] if z_max == None or z_max > self.Z_range[-1] else z_max
        n_min = self.N_range[0] if n_min == None or n_min < self.N_range[0] else n_min
        n_max = self.N_range[-1] if n_max == None or n_max > self.N_range[-1] else n_max
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        # prepare the data for displaying
        for Z in Z_range:  
            self.cur.execute("SELECT N,MASSEXC,ELEMENT FROM MASSEXCDATA WHERE Z=? AND N>=? AND N<=? ORDER BY N", (int(Z), int(n_min), int(n_max)))
            result = np.array([[*row] for row in self.cur.fetchall()]).T
            temp_massexc, temp_element = np.zeros_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
            if len(result)>0: temp_massexc[result[0].astype(int)-n_min] = result[1].astype(np.float64)
            if len(result)>0: temp_element[result[0].astype(int)-n_min] = result[2]
            temp_massexc.reshape(1,len(N_range))
            temp_element.reshape(1,len(N_range))
            if Z == Z_range[0]:
                data = temp_massexc
                data_annote = temp_element
            else:
                data = np.vstack((data, temp_massexc))
                data_annote = np.vstack((data_annote, temp_element))
        # drop table massexcdata
        self.cur.execute("DROP TABLE MASSEXCDATA")
        self.conn.commit()
        # setup the plot
        norm = colors.TwoSlopeNorm(vcenter=0)
        fig, ax = plt.subplots()
        im = self.heatmap_colorbar(data, ax=ax, xy_ticks={'x_ticks':[], 'y_ticks':[], 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in N_range], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in Z_range], 'xlabel':'N', 'ylabel':'Z'}, cbar_ticklabels=[], cbar_label='MASS EXCESS (keV)', cbar_kw=dict(extend='both', orientation='vertical'), norm=norm, cmap='RdBu_r')
        if add_sheline: self.add_shellline(ax)
        plt.show()

    def plot_decay_mode(self, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True):
        '''
        plot the decay mode chart
        only contains simple decay, e.g., alpha emission, proton emission, neutron emission, beta- decay, beta+ decay, spontaneous fission.
        only contains measured and no isomers.
        '''
        time_unit = {'Yy': 31536000*1e24, 'Zy': 31536000*1e21, 'Ey': 31536000*1e18, 'Py': 31536000*1e15, 'Ty': 31536000*1e12, 'Gy': 31536000*1e9, 'My': 31536000*1e6, 'ky': 31536000*1e3, 'y': 31536000, 'd': 86400, 'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18, 'zs': 1e-21, 'ys': 1e-24}
        #self.cur.execute("DROP TABLE DECAYDATA")
        # create table decaydata
        self.cur.execute("CREATE TABLE DECAYDATA (ELEMENT CHAR(2), Z INT, N INT, DECAYMODE TEXT);")
        self.conn.commit()
        result = self.cur.execute("SELECT A, ELEMENT, Z, N, DECAYMODE, HALFLIFE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = []
        colorNumSelect = {"B+": 2, "B-": 3, "EC": 4, "A": 5, "SF": 6, "n": 7, "p": 8, "IS": 9}
        for row in result:
            temp = row[-2].split(";")
            if row[-1] == 'stbl':
                re_set.append((*row[1:-2], 9))
            else:
                for item in temp:
                    if '=' in item or '~' in item:
                        if decayModeSelect(item) == "IS":
                            if float(row[-1].split()[0]) * time_unit[row[-1].split()[-1]] > .7e9 * time_unit['y']:
                                re_set.append((*row[1:-2], colorNumSelect[decayModeSelect(item)]))
                                break
                            else:
                                continue
                        else:
                            if decayModeSelect(item) == None:
                                pass
                            else:
                                re_set.append((*row[1:-2], colorNumSelect[decayModeSelect(item)]))
                            break
        self.cur.executemany("INSERT INTO DECAYDATA (ELEMENT, Z, N, DECAYMODE) VALUES (?,?,?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] if z_min == None or z_min < self.Z_range[0] else z_min
        z_max = self.Z_range[-1] if z_max == None or z_max > self.Z_range[-1] else z_max
        n_min = self.N_range[0] if n_min == None or n_min < self.N_range[0] else n_min
        n_max = self.N_range[-1] if n_max == None or n_max > self.N_range[-1] else n_max
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        for Z in Z_range:
            self.cur.execute("SELECT N,DECAYMODE,ELEMENT FROM DECAYDATA WHERE Z=? AND N>=? AND N<=? ORDER BY N", (int(Z), int(n_min), int(n_max)))
            result = np.array([[*row] for row in self.cur.fetchall()]).T
            temp_decay, temp_element = np.ones_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
            if len(result)>0: temp_decay[result[0].astype(int)] = result[1].astype(np.float64)
            if len(result)>0: temp_element[result[0].astype(int)] = result[2]
            temp_decay.reshape(1,len(N_range))
            temp_element.reshape(1,len(N_range))
            if Z == Z_range[0]:
                data = temp_decay
                data_annote = temp_element
            else:
                data = np.vstack((data, temp_decay))
                data_annote = np.vstack((data_annote, temp_element))
        data = np.ma.masked_where(data<=1., data)
        data_annote = np.ma.masked_where(data_annote=='  ', data_annote)
        # drop table lifedata
        self.cur.execute("DROP TABLE DECAYDATA")
        self.conn.commit()
        # setup the plot
        decay_modes = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        bounds = np.array(decay_modes)
        #mpl.rcParams['axes.labelsize'] = 24
        #mpl.rcParams['xtick.labelsize'] = 20
        #mpl.rcParams['ytick.labelsize'] = 20
        fig, ax = plt.subplots()
        self.add_drip_line(ax=ax)
        ## heatmap - Chinese printed (binary)
        #mpl.rcParams['hatch.linewidth'] = 1.
        #decay_modes_ticklabels = [r'$\beta +$'+'衰变', r'$\beta -$'+'衰变', '轨道电子俘获', r'$\alpha$'+'衰变', '自发裂变', '中子发射', '质子发射', '稳定核素']
        #self.heatmap_hatch(data, ax=ax, bounds=bounds, xy_ticks={'x_ticks':np.arange(0, N_range.max()+1, step=5), 'y_ticks':np.arange(0, Z_range.max()+1, step=5), 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in np.arange(0, N_range.max()+1, step=5)], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in np.arange(0, Z_range.max()+1, step=5)], 'xlabel':'中子数', 'ylabel':'质子数'}, hatch_list=['o', 'xx', '++++', '**', '....', '//////', '\\\\\\\\\\', 'OO'], hatch_ticklabels=decay_modes_ticklabels, legend_title='衰变模式', legend_kw=dict(loc='lower right', ncol=2, fontsize=24, title_fontsize=24), cbar_bin={'cbar_max': 130, 'cbar_range': [87, 72, 40, 50, 60, 30, 20, 129]}, chinese=True)
        # heatmap - color
        decay_modes_ticklabels = [r'$\beta +$', r'$\beta -$', 'EC', r'$\alpha$', 'SF', 'n', 'p', 'stable']
        norm = colors.BoundaryNorm(bounds, ncolors=8)
        cmap = colors.ListedColormap((*plt.cm.get_cmap('tab10').colors[:7], (0,0,0)))
        patch_list = [mpatches.Patch(color=cmap(b)) for b in range(len(bounds))]
        ## heatmap - Chinese printed (color)
        #im = self.heatmap_legend(data, ax=ax, xy_ticks={'x_ticks':np.arange(0, N_range.max()+1, step=5), 'y_ticks':np.arange(0, Z_range.max()+1, step=5), 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in np.arange(0, N_range.max()+1, step=5)], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in np.arange(0, Z_range.max()+1, step=5)], 'xlabel':'中子数', 'ylabel':'质子数'}, patch_list=patch_list, legend_ticklabels=decay_modes_ticklabels, legend_title='衰变模式', legend_kw=dict(loc='lower right', ncol=2, fontsize=22, title_fontsize=22), chinese=True, norm=norm, cmap=cmap)
        # heatmap - original
        im = self.heatmap_legend(data, ax=ax, xy_ticks={'x_ticks':np.arange(0, N_range.max()+1, step=5), 'y_ticks':np.arange(0, Z_range.max()+1, step=5), 'x_ticklabels':['{:}'.format(n) if n%10==0 else '' for n in np.arange(0, N_range.max()+1, step=5)], 'y_ticklabels':['{:}'.format(z) if z%10==0 else '' for z in np.arange(0, Z_range.max()+1, step=5)], 'xlabel':'N', 'ylabel':'Z'}, patch_list=patch_list, legend_ticklabels=decay_modes_ticklabels, legend_title=r'$\bf{DECAY}$ '+r' $\bf{MODE}$', legend_kw=dict(loc='lower right', ncol=2, fontsize=20, title_fontsize=20), norm=norm, cmap=cmap)

        #self.plot_blank_chart(include_estimated=True, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=add_sheline, embedded=True, ax=ax)
        if add_sheline: self.add_shellline(ax=ax)
        plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)
        plt.show()

    def plot_BrhoRegion(self, Brho_range, gamma_t, add_sheline=True):
        '''
        plot the Brho Region depend on gamma_t 
        '''
        #self.cur.execute("DROP TABLE DECAYDATA")
        Brho_min, Brho_max = Brho_range[0], Brho_range[-1]
        c = 299792458 # speed of light in m/s
        e = 1.602176634e-19 # elementary charge in C
        me = 5.48579909065e-4 # electron mass in u
        u2kg = 1.66053906660e-27 # amount of kg per 1 u
        MeV2u = 1.07354410233e-3 # amount of u per 1 MeV
        time_unit = {'Yy': 31536000*1e24, 'Zy': 31536000*1e21, 'Ey': 31536000*1e18, 'Py': 31536000*1e15, 'Ty': 31536000*1e12, 'Gy': 31536000*1e9, 'My': 31536000*1e6, 'ky': 31536000*1e3, 'y': 31536000, 'd': 86400, 'h': 3600, 'm': 60, 's': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18, 'zs': 1e-21, 'ys': 1e-24}
        self.cur.execute("CREATE TABLE DECAYDATA (ELEMENT CHAR(2), Z INT, N INT, DECAYMODE TEXT);")
        self.conn.commit()
        result = self.cur.execute("SELECT A, ELEMENT, Z, N, DECAYMODE, HALFLIFE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = []
        # add stability line
        for row in result:
            temp = row[-2].split(";")
            if row[-1] == 'stbl':
                re_set.append((*row[1:-2], 2))
            else:
                for item in temp:
                    if '=' in item or '~' in item:
                        if decayModeSelect(item) == "IS":
                            if float(row[-1].split()[0]) * time_unit[row[-1].split()[-1]] > .7e9 * time_unit['y']:
                                re_set.append((*row[1:-2], 2))
                                break
                            else:
                                continue
                        else:
                            if decayModeSelect(item) == None:
                                pass
                            else:
                                re_set.append((*row[1:-2], 1))
                            break
        self.cur.executemany("INSERT INTO DECAYDATA (ELEMENT, Z, N, DECAYMODE) VALUES (?,?,?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] 
        z_max = self.Z_range[-1]
        n_min = self.N_range[0]
        n_max = self.N_range[-1]
        N_range = np.arange(n_min, n_max+1)
        Z_range = np.arange(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        for Z in Z_range:
            self.cur.execute("SELECT N,DECAYMODE,ELEMENT FROM DECAYDATA WHERE Z=? AND N>=? AND N<=? ORDER BY N", (int(Z), int(n_min), int(n_max)))
            result = np.array([[*row] for row in self.cur.fetchall()]).T
            temp_decay, temp_element = np.ones_like(N_range).astype(np.float64), np.array(['  ' for i in N_range])
            if len(result)>0: temp_decay[result[0].astype(int)] = result[1].astype(np.float64)
            if len(result)>0: temp_element[result[0].astype(int)] = result[2]
            temp_decay.reshape(1,len(N_range))
            temp_element.reshape(1,len(N_range))
            if Z == Z_range[0]:
                data = temp_decay
            else:
                data = np.vstack((data, temp_decay))
        #data = np.ma.masked_where(data>=0, data)
        fig, ax = plt.subplots()
        ax.pcolormesh(self.x, self.y, data, cmap='binary',rasterized=True)
        self.cur.execute("DROP TABLE DECAYDATA")
        self.conn.commit()
        # plot Brho region
        const_temp =  c * np.sqrt(gamma_t**2 - 1) * u2kg / e
        x_range = np.linspace(0, 220, 500)
        import matplotlib.patches as pat
        w_bare = pat.Wedge(center=(0,0), r=220, theta1=np.degrees(np.arctan(const_temp/(Brho_max-const_temp))), theta2=np.degrees(np.arctan(const_temp/(Brho_min-const_temp))), color='royalblue', alpha=0.3)
        w_Hlike = pat.Wedge(center=(-1,1), r=210, theta1=np.degrees(np.arctan(const_temp/(Brho_max-const_temp))), theta2=np.degrees(np.arctan(const_temp/(Brho_min-const_temp))), color='darkorange', alpha=0.3)
        w_Helike = pat.Wedge(center=(-2,2), r=200, theta1=np.degrees(np.arctan(const_temp/(Brho_max-const_temp))), theta2=np.degrees(np.arctan(const_temp/(Brho_min-const_temp))), color='green', alpha=0.3)
        ax.add_patch(w_bare)
        ax.add_patch(w_Hlike)
        ax.add_patch(w_Helike)
        ax.set_xlim([-10, 220])
        ax.set_ylim([-10, 200])
        ax.axis('off')
        ax.annotate("N", xy=(100, -1), xytext=(-100,0), textcoords='offset points', size=20, arrowprops=dict(facecolor='gray', shrink=0.05), horizontalalignment='center', verticalalignment='center')
        ax.annotate("Z", xy=(-1, 100), xytext=(0,-100), textcoords='offset points', size=20, arrowprops=dict(facecolor='gray', shrink=0.05), horizontalalignment='center', verticalalignment='center')
        legend_elements=[lines.Line2D([0],[0], marker='o', color='w', label='Bare', markerfacecolor='royalblue', markersize=12, alpha=0.3), 
                lines.Line2D([0],[0], marker='o', color='w', label='H-like', markerfacecolor='darkorange', markersize=12, alpha=0.3),
                lines.Line2D([0],[0], marker='o', color='w', label='He-like', markerfacecolor='green', markersize=12, alpha=0.3)]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=18, frameon=False)
        self.plot_blank_chart(include_estimated=True, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True, embedded=True, ax=ax)
        plt.show()



    def plot_blank_chart(self, include_estimated=True, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True, embedded=False, ax=None):
        '''
        plot the blank nuclide chart

        include_estimated:  True for including the nuclides with the estimated mass excess. False for not including.
        '''
        #self.cur.execute("DROP TABLE ELEMENTDATA")
        # create table massaxcdata
        self.cur.execute("CREATE TABLE ELEMENTDATA (Z INT, N INT);")
        self.conn.commit()
        result = self.cur.execute("SELECT ELEMENT, Z, N, A, MASS, SOURCE FROM IONICDATA WHERE TYPE='bare' AND ISOMERIC='0'").fetchall()
        re_set = [(row[1:3]) for row in result] if include_estimated else [(row[1:3]) for row in result if row[-1] == 'measured']
        self.cur.executemany("INSERT INTO ELEMENTDATA (Z, N) VALUES (?,?)", re_set)
        self.conn.commit()
        z_min = self.Z_range[0] if z_min == None or z_min < self.Z_range[0] else z_min
        z_max = self.Z_range[-1] if z_max == None or z_max > self.Z_range[-1] else z_max
        n_min = self.N_range[0] if n_min == None or n_min < self.N_range[0] else n_min
        n_max = self.N_range[-1] if n_max == None or n_max > self.N_range[-1] else n_max
        N_range = range(n_min, n_max+1)
        Z_range = range(z_min, z_max+1)
        self.x, self.y = np.arange(n_min-0.5,n_max+1.5), np.arange(z_min-0.5, z_max+1.5)
        # prepare the data for displaying
        _N = [[j[0] if len(i)>0 else None for j in i] for i in [self.cur.execute("SELECT N FROM ELEMENTDATA WHERE Z=?", (Z,)).fetchall() for Z in Z_range]]
        h_line, hd_line = [], []
        for i in range(len(Z_range)):
            if len(_N[i])>0:
                if _N[i] == range(_N[i][0], _N[i][-1]+1):
                    h_line.append(([_N[i][0]-0.5, _N[i][-1]+0.5], [Z_range[i]+0.5, Z_range[i]+0.5]))
                    h_line.append(([_N[i][0]-0.5, _N[i][-1]+0.5], [Z_range[i]-0.5, Z_range[i]-0.5]))
                else:
                    for j in _N[i]:
                        hd_line.append(([j-0.5, j+0.5], [Z_range[i]+0.5, Z_range[i]+0.5]))
                        hd_line.append(([j-0.5, j+0.5], [Z_range[i]-0.5, Z_range[i]-0.5]))
        _Z = [[j[0] if len(i)>0 else None for j in i] for i in [self.cur.execute("SELECT Z FROM ELEMENTDATA WHERE N=?", (N,)).fetchall() for N in N_range]]
        v_line, vd_line = [], []
        for i in range(len(N_range)):
            if len(_Z[i])>0:
                if _Z[i] == range(_Z[i][0], _Z[i][-1]+1):
                    v_line.append(([N_range[i]+0.5, N_range[i]+0.5], [_Z[i][0]-0.5, _Z[i][-1]+0.5]))
                    v_line.append(([N_range[i]-0.5, N_range[i]-0.5], [_Z[i][0]-0.5, _Z[i][-1]+0.5]))
                else:
                    for j in _Z[i]:
                        vd_line.append(([N_range[i]+0.5, N_range[i]+0.5], [j-0.5, j+0.5]))
                        vd_line.append(([N_range[i]-0.5, N_range[i]-0.5], [j-0.5, j+0.5]))
        # setup the plot
        if not embedded:
            fig, ax = plt.subplots()
            ax.set_xlim((n_min-10,n_max+10))
            ax.set_ylim((z_min-10,z_max+10))
        for x, y in h_line + hd_line + v_line + vd_line:#hu_line + hp_line + vr_line + vl_line:
            line = lines.Line2D(x, y, color='lightgray', lw=1, axes=ax)
            ax.add_line(line)
        # drop table massexcdata
        self.cur.execute("DROP TABLE ELEMENTDATA")
        self.conn.commit()
        if add_sheline: self.add_shellline(ax)
        if not embedded: plt.show()

    def add_shellline(self, ax):
        '''
        adding the line of shell

        ax:         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
        '''
        magic_num_proton = [2, 8, 20, 28, 50, 82, 114]
        magic_num_neutron = [2, 8, 20, 28, 50, 82, 126, 184]
        h_result  = [self.cur.execute("SELECT min(N), max(N) FROM IONICDATA WHERE Z=?", (Z,)).fetchall()[0] for Z in magic_num_proton]
        hu_line = [([h_result[i][0]-6.5, h_result[i][1]+8.5], [magic_num_proton[i]+0.5, magic_num_proton[i]+0.5]) for i in range(len(magic_num_proton[:-1]))]
        hd_line = [([h_result[i][0]-6.5, h_result[i][1]+8.5], [magic_num_proton[i]-0.5, magic_num_proton[i]-0.5]) for i in range(len(magic_num_proton[:-1]))]
        v_result  = [self.cur.execute("SELECT min(Z), max(Z) FROM IONICDATA WHERE N=?", (N,)).fetchall()[0] for N in magic_num_neutron[:-1]]
        vr_line = [([magic_num_neutron[i]+0.5, magic_num_neutron[i]+0.5], [v_result[i][0]-4.5, v_result[i][1]+6.5]) for i in range(len(magic_num_neutron[:-1]))]
        vl_line = [([magic_num_neutron[i]-0.5, magic_num_neutron[i]-0.5], [v_result[i][0]-4.5, v_result[i][1]+6.5]) for i in range(len(magic_num_neutron[:-1]))]
        for x, y in hu_line + hd_line + vr_line + vl_line:
            line = lines.Line2D(x, y, color='gray', axes=ax)
            #line = lines.Line2D(x, y, color='black', axes=ax)
            ax.add_line(line)
        
    def add_drip_line(self, ax):
        '''
        adding the drip line based on N. Wang's estimated separation energy data

        ax:         A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
        '''
        file_path = './separate_energy/'
        file_names = ['NWang_WS4.S2n.s2n', 'NWang_WS4.S2p.s2p', 'NWang_WS4.Sn.sn', 'NWang_WS4.Sp.sp']
        #self.cur.execute("DROP TABLE ESTIMATEDDATA")
        self.cur.execute("CREATE TABLE ESTIMATEDDATA (N INT, Z INT);")
        self.conn.commit()
        for _file in file_names:
            with open(file_path+_file) as _f:
                re_set = [(int(l.split()[0]), int(l.split()[1])) for l in _f if float(l.split()[-1]) > 0]
            self.cur.executemany("INSERT INTO ESTIMATEDDATA (N, Z) VALUES (?,?)", re_set)
            self.conn.commit()
        result = self.cur.execute("SELECT N, Z FROM ESTIMATEDDATA GROUP BY N, Z HAVING COUNT(*) == 4").fetchall()
        N_range = np.array([item[0] for item in result])
        Z_range = np.array([item[1] for item in result])
        data = coo_matrix((np.ones_like(N_range), (Z_range, N_range)), shape=(Z_range.max()+1, N_range.max()+1)).toarray()
        self.cur.execute("DROP TABLE ESTIMATEDDATA")
        ax.pcolormesh(np.arange(-0.5, N_range.max()+1.5), np.arange(-0.5,Z_range.max()+1.5), data, cmap='binary', alpha=0.1)

    def plot_additional_data(self, data={'data_file':'', 'sep':"\s+"}, patch_paras=[0, 10, 5], set_cmap='magma_r', include_estimated=False, add_sheline=True):
        '''
        plot heatmap for the optional data file

        data:           data file to be utilized (.csv, .txt, .dat, etc.) with format of
                        Z   N   VALUE
                        1   1   10
                        1   2   100
                        ...
                        'data_file'='./data.txt', sep='\s+'
        patch_paras:    [start, end, number] of the patch
        set_cmap:       colormap for the heatmap
        '''
        data_temp = pd.read_csv(data['data_file'], sep=data['sep'], index_col=False, usecols=[0, 1, 2], names=['Z', 'N', 'VALUE'])
        z_min, z_max = min(data_temp['Z'].values), max(data_temp['Z'].values)
        n_min, n_max = min(data_temp['N'].values), max(data_temp['N'].values)
        value = np.full((z_max-z_min+1,n_max-n_min+1), np.nan, dtype=np.float64)
        for i in range(len(data_temp)):
            value[data_temp['Z'][i]-z_min, data_temp['N'][i]-n_min] = data_temp['VALUE'][i]
        Z, N = np.arange(z_min-0.5,z_max+1.5), np.arange(n_min-0.5,n_max+1.5)
        fig, ax = plt.subplots()
        self.plot_blank_chart(include_estimated=include_estimated, z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=add_sheline, embedded=True, ax=ax)
        bounds = np.linspace(patch_paras[0], patch_paras[1], patch_paras[2]+1)
        limits_ticklabels = ["{:}-{:}".format(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        cmap = plt.cm.get_cmap(set_cmap, len(limits_ticklabels))
        patch_list = [mpatches.Patch(color=cmap(b)) for b in range(len(bounds))]
        ax.pcolormesh(N, Z, value, norm=norm, cmap=set_cmap)
        ax.legend(patch_list, limits_ticklabels) 
        plt.show()


if __name__ == '__main__':
    pproj = plot_nubase()
    #pproj.plot_yield(include_estimated=True, add_sheline=True)
    #pproj.plot_BrhoRegion(Brho_range=[5,9.4], gamma_t=1.359)
    #pproj.plot_mass_excess(include_estimated=True, z_min=10, z_max=20, n_min=0, n_max=50, add_sheline=True)
    pproj.plot_blank_chart(include_estimated=False, z_min=1, z_max=118, n_min=0, n_max=200, add_sheline=True)
    #pproj.plot_mass_accuracy(z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True)
    #pproj.plot_life(z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True)
    #pproj.plot_decay_mode(z_min=None, z_max=None, n_min=None, n_max=None, add_sheline=True)
    #pproj.plot_additional_data(data={'data_file':'C:/Users/Van/Desktop/analysis/First0+.txt', 'sep':"\s+"}, patch_paras=[100, 20000, 5], set_cmap='magma_r', include_estimated=False, add_sheline=True)
