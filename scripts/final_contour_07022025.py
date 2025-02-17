# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:49:15 2025

@author: Jevare
"""


from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import glob
import os

#---------------------------------

def masking(input_Pearson,input_pvalue):
    
    data_model_Pearson_new = np.flipud(input_Pearson)
    
    mask1 = np.zeros_like(data_model_Pearson_new, dtype=bool)
    mask1[np.tril_indices_from(mask1)] = True
    
    mask2 = np.ma.masked_where(np.flipud(input_pvalue.astype(float)) < 0.05, np.flipud(input_pvalue).astype(float))
    mask2[np.triu_indices_from(mask2, k=1)] = True 
    mask2 = np.flipud(mask2)
    
    mask1_new = np.zeros((91,91))
    mask2_new = np.zeros((91,91))
    
    data_model_pvalue_new = np.flipud(input_pvalue)
    
    for i in range(91):
        for j in range(91):
            if mask1[i][j] == True:
                mask1_new[i,j] = data_model_Pearson_new[i][j]
            if mask1[i][j] == False:
                mask1_new[i,j] = np.nan            
    
    for i in range(91):
        for j in range(91):
            if mask2.mask[i][j] == False:
                mask2_new[i,j] = data_model_pvalue_new[i][j]
            elif mask2.mask[i][j] == True:
                mask2_new[i,j] = np.nan 
               
    for i in range(91):
        for j in range(91):
            if mask2.data[i][j] == 1:
                mask2_new[i,j] = np.nan     
                
    return data_model_Pearson_new, data_model_pvalue_new, mask1_new, mask2_new

#------------------------------------
#preparation colorbar
    
import matplotlib.colors as mcolors
cmap = 'autumn_r'
#cmap = ListedColormap(['yellow', 'gold','orange', 'darkorange','xkcd:blood orange','xkcd:tomato red','xkcd:scarlet','darkred','brown','black'])
color1 = cm.get_cmap(cmap)
color1a = color1(np.linspace(0, 1, 12))

cmap_oth = 'YlGnBu_r'
color2 = cm.get_cmap(cmap_oth)
color2a = color2(np.linspace(0, 1, 12))


comb_cmap = np.vstack((color2a, color1a))

mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', comb_cmap)

newcmp = mymap
newcmp1 = cmap

#-----------------------------------------------

k_lev = '530'

model = 'MERRA2' #ERA5 or MERRA2 or EL_ERA5
print('model: '+str(model)+'')

path_in = 'C:/...' #location, where your data is stored
path_out = 'C:/...' #location of your output folder

     
if model == 'EL_ERA5':
    path = ''+str(path_in)+'' 
    all_files_pearson = glob.glob(os.path.join(path, 'new_'+str(k_lev)+'_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'new_'+str(k_lev)+'_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)    


    #all_files_pearson[0][86:96] #apsc30hPa
    #all_files_pearson[1][86:96] #apsc50hPa
    #all_files_pearson[2][86:95] #m_strat_T
    #all_files_pearson[3][86:94] #oz30hPa
    #all_files_pearson[4][86:94] #oz50hPa
    #all_files_pearson[5][86:89] #tco
    #all_files_pearson[6][86:103] #temp30hPa
    #all_files_pearson[7][86:103] #temp50hPa
    #all_files_pearson[8][86:90] #vpsc
    #
    #all_files_pearson[0][97:105] #oz30hPa
    #all_files_pearson[1][97:105] #oz50hPa
    #all_files_pearson[2][96:99] #tco
    #all_files_pearson[3][95:103] #oz30hPa
    #all_files_pearson[4][95:103] #oz50hPa
    #all_files_pearson[5][90:93] #tco
    #all_files_pearson[6][86:103] #oz30hPa
    #all_files_pearson[7][86:103] #oz50hPa
    #all_files_pearson[8][91:94] #tco

    
    var_temp = [all_files_pearson[0][86:96],all_files_pearson[1][86:96],
                all_files_pearson[2][86:95],all_files_pearson[3][86:94],
                all_files_pearson[4][86:94],all_files_pearson[5][86:89],
                all_files_pearson[6][86:103],all_files_pearson[7][86:103],
                all_files_pearson[8][86:90]]
    
    var_temp_txt = ['Apsc30','Apsc50',
                    'Tstrat',
                    'O$_{3}$30','O$_{3}$50',
                    'TCO',
                    'T30','T50',
                    'Vpsc']
    
    var_ozone = [all_files_pearson[0][97:105],all_files_pearson[1][97:105],
                 all_files_pearson[2][96:99],all_files_pearson[3][95:103],
                 all_files_pearson[4][95:103],all_files_pearson[5][90:93],
                 all_files_pearson[6][104:112],all_files_pearson[7][104:112],
                 all_files_pearson[8][91:94]]
    
    var_ozone_txt = ['O$_{3}$30','O$_{3}$50',
                     'TCO',
                     'O$_{3}$30','O$_{3}$50',
                     'TCO',
                     'O$_{3}$30','O$_{3}$50',
                     'TCO']
    
    
    len_vartemp = len(var_temp)
    
    #preparation:
    len_ylab = np.shape(concatenated_df_pearson.iloc[:,0])[0]
    ylab_new = []
    for l in range(len_ylab):
        ylab_new.append(str(np.ravel(concatenated_df_pearson.iloc[:,0])[l]))
    
    
    xlab = []
    for k in range(1,len_ylab+1):
        xlab.append(concatenated_df_pearson.columns[k])
        
        
    n = np.size(xlab)+1 # drop the index column and choose just every variable columns e.g. above
    val_pearson = []
    val_pvalue = []
    
    for h in range(len_vartemp):
        for i in range(0,len_ylab):
            for j in range(1,len_ylab+1):
                val_pearson.append(concatenated_df_pearson.iloc[i,j+(h*n)])
                val_pvalue.append(concatenated_df_pvalue.iloc[i,j+(h*n)])            
    
    val_pearson_new = np.reshape(val_pearson,(len_vartemp,len_ylab,len_ylab))
    
    val_pvalue_new = np.reshape(val_pvalue,(len_vartemp,len_ylab,len_ylab))
    
    
    #final_data_pearson = np.zeros((9,91,91))
    #final_data_pvalue = np.zeros((9,91,91))
    #
    #
    #for m in range(9):
    #    for n in range(91):
    #        for o in range(91):
    #            final_data_pearson[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[2][n,o]
    #            final_data_pvalue[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[3][n,o]
    #            print(''+str(m)+'-'+str(n)+'-'+str(o)+'')
    
    
    
    final_data0 = masking(val_pearson_new[0],val_pvalue_new[0])
    final_data1 = masking(val_pearson_new[1],val_pvalue_new[1])
    final_data2 = masking(val_pearson_new[2],val_pvalue_new[2])
    final_data3 = masking(val_pearson_new[3],val_pvalue_new[3])
    final_data4 = masking(val_pearson_new[4],val_pvalue_new[4])
    final_data5 = masking(val_pearson_new[5],val_pvalue_new[5])
    final_data6 = masking(val_pearson_new[6],val_pvalue_new[6])
    final_data7 = masking(val_pearson_new[7],val_pvalue_new[7])
    final_data8 = masking(val_pearson_new[8],val_pvalue_new[8])

    
    #plots
    fig = plt.figure(1, figsize=(20,30))  
    fig.add_subplot(321)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
     
    ax.contourf(np.flipud(final_data7[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'EL-$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=20,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)

    ax.contourf(np.flipud(final_data1[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)
        
    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=20,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')

    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
        
    
    plt.savefig(r''+str(path_out)+'quad6_diff_lev_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'EL-$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 18)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=20,color = 'k',bbox= props1) 
     
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
       
    
    
    plt.savefig(r''+str(path_out)+'quad4_diff_lev_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 




elif model == 'MERRA2':
    path = ''+str(path_in)+'Excel2_'+str(model)+'/'  
    all_files_pearson = glob.glob(os.path.join(path, 'final_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'final_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)    


    #all_files_pearson[0][74:84] #apsc30hPa
    #all_files_pearson[1][74:84] #apsc50hPa
    #all_files_pearson[2][74:83] #m_strat_T
    #all_files_pearson[3][74:82] #oz30hPa
    #all_files_pearson[4][74:82] #oz50hPa
    #all_files_pearson[5][74:91] #temp30hPa
    #all_files_pearson[6][74:91] #temp50hPa
    #all_files_pearson[7][74:77] #tco
    #all_files_pearson[8][74:78] #vpsc
    #all_files_pearson[9][74:76] #vT
    #                   
    #all_files_pearson[0][85:93] #oz30hPa
    #all_files_pearson[1][85:93] #oz50hPa
    #all_files_pearson[2][84:87] #tco
    #all_files_pearson[3][83:91] #oz30hPa
    #all_files_pearson[4][83:91] #oz50hPa
    #all_files_pearson[5][92:100] #oz30hPa
    #all_files_pearson[6][92:100] #oz50hPa
    #all_files_pearson[7][78:81] #tco
    #all_files_pearson[8][79:82] #tco
    #all_files_pearson[9][77:80] #tco

    
    var_temp = [all_files_pearson[0][74:84],all_files_pearson[1][74:84],
                all_files_pearson[2][74:83],all_files_pearson[3][74:82],
                all_files_pearson[4][74:82],all_files_pearson[5][74:91],
                all_files_pearson[6][74:91],all_files_pearson[7][74:77],
                all_files_pearson[8][74:78],all_files_pearson[9][74:76]]                  
                
                
    var_temp_txt = ['Apsc30','Apsc50',
                    'Tstrat',
                    'O$_{3}$30','O$_{3}$50',
                    'T30','T50',
                    'TCO','Vpsc','EHF']
    
    var_ozone = [all_files_pearson[0][85:93],all_files_pearson[1][85:93],
                 all_files_pearson[2][84:87],all_files_pearson[3][83:91],
                 all_files_pearson[4][83:91],all_files_pearson[5][92:100],
                 all_files_pearson[6][92:100],all_files_pearson[7][78:81],
                 all_files_pearson[8][79:82],all_files_pearson[9][77:80]]
    
    var_ozone_txt = ['O$_{3}$30','O$_{3}$50',
                     'TCO',
                     'O$_{3}$30','O$_{3}$50',
                     'O$_{3}$30','O$_{3}$50',
                     'TCO','TCO','TCO']
    
    
    len_vartemp = len(var_temp)
    
    #preparation:
    len_ylab = np.shape(concatenated_df_pearson.iloc[:,0])[0]
    ylab_new = []
    for l in range(len_ylab):
        ylab_new.append(str(np.ravel(concatenated_df_pearson.iloc[:,0])[l]))
    
    
    xlab = []
    for k in range(1,len_ylab+1):
        xlab.append(concatenated_df_pearson.columns[k])
        
        
    n = np.size(xlab)+1 # drop the index column and choose just every variable columns e.g. above
    val_pearson = []
    val_pvalue = []
    
    for h in range(len_vartemp):
        for i in range(0,len_ylab):
            for j in range(1,len_ylab+1):
                val_pearson.append(concatenated_df_pearson.iloc[i,j+(h*n)])
                val_pvalue.append(concatenated_df_pvalue.iloc[i,j+(h*n)])            
    
    val_pearson_new = np.reshape(val_pearson,(len_vartemp,len_ylab,len_ylab))
    
    val_pvalue_new = np.reshape(val_pvalue,(len_vartemp,len_ylab,len_ylab))
        
    
    #final_data_pearson = np.zeros((9,91,91))
    #final_data_pvalue = np.zeros((9,91,91))
    #
    #
    #for m in range(9):
    #    for n in range(91):
    #        for o in range(91):
    #            final_data_pearson[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[2][n,o]
    #            final_data_pvalue[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[3][n,o]
    #            print(''+str(m)+'-'+str(n)+'-'+str(o)+'')
    
    
    
    final_data0 = masking(val_pearson_new[0],val_pvalue_new[0])
    final_data1 = masking(val_pearson_new[1],val_pvalue_new[1])
    final_data2 = masking(val_pearson_new[2],val_pvalue_new[2])
    final_data3 = masking(val_pearson_new[3],val_pvalue_new[3])
    final_data4 = masking(val_pearson_new[4],val_pvalue_new[4])
    final_data5 = masking(val_pearson_new[5],val_pvalue_new[5])
    final_data6 = masking(val_pearson_new[6],val_pvalue_new[6])
    final_data7 = masking(val_pearson_new[7],val_pvalue_new[7])
    final_data8 = masking(val_pearson_new[8],val_pvalue_new[8])
    final_data9 = masking(val_pearson_new[9],val_pvalue_new[9])
    
    #plots
    fig = plt.figure(1, figsize=(20,30))  
    fig.add_subplot(321)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'$\bf{MERRA2}$' '\n' '(a) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=20,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data1[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=20,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    #cbar.ax.invert_xaxis()
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
        
    
    plt.savefig(r''+str(path_out)+'quad6_diff_lev_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'$\bf{MERRA2}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 18)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data7[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=20,color = 'k',bbox= props1) 
     


    fig.add_subplot(224)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data9[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data9[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(d) '+str(var_temp_txt[9])+' vs. '+str(var_ozone_txt[9])+'',fontsize=20,color = 'k',bbox= props1) 

    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
       
    
    
    plt.savefig(r''+str(path_out)+'quad4_diff_lev_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 




elif model == 'ERA5':
    path = ''+str(path_in)+'Excel2_'+str(model)+'/'  
    all_files_pearson = glob.glob(os.path.join(path, 'final_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'final_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)    


    #all_files_pearson[0][74:84] #apsc30hPa
    #all_files_pearson[1][74:84] #apsc50hPa
    #all_files_pearson[2][74:83] #m_strat_T
    #all_files_pearson[3][74:82] #oz30hPa
    #all_files_pearson[4][74:82] #oz50hPa
    #all_files_pearson[5][74:91] #temp30hPa
    #all_files_pearson[6][74:91] #temp50hPa
    #all_files_pearson[7][74:77] #tco
    #all_files_pearson[8][74:78] #vpsc
    #all_files_pearson[9][74:76] #vT
    #            
    #all_files_pearson[0][85:93] #oz30hPa
    #all_files_pearson[1][85:93] #oz50hPa
    #all_files_pearson[2][84:87] #tco
    #all_files_pearson[3][83:91] #oz30hPa
    #all_files_pearson[4][83:91] #oz50hPa
    #all_files_pearson[5][92:100] #oz30hPa
    #all_files_pearson[6][92:100] #oz50hPa
    #all_files_pearson[7][78:81] #tco
    #all_files_pearson[8][79:82] #tco
    #all_files_pearson[9][77:80] #tco

    
    var_temp = [all_files_pearson[0][70:80],all_files_pearson[1][70:80],
                all_files_pearson[2][70:79],all_files_pearson[3][70:78],
                all_files_pearson[4][70:78],all_files_pearson[5][70:87],
                all_files_pearson[6][70:87],all_files_pearson[7][70:73],
                all_files_pearson[8][70:74],all_files_pearson[9][70:72]]                  
                
                
    var_temp_txt = ['Apsc30','Apsc50',
                    'Tstrat',
                    'O$_{3}$30','O$_{3}$50',
                    'T30','T50',
                    'TCO','Vpsc','EHF']
    
    var_ozone = [all_files_pearson[0][81:89],all_files_pearson[1][81:89],
                 all_files_pearson[2][80:83],all_files_pearson[3][79:87],
                 all_files_pearson[4][79:87],all_files_pearson[5][88:96],
                 all_files_pearson[6][88:96],all_files_pearson[7][74:77],
                 all_files_pearson[8][75:78],all_files_pearson[9][73:76]]
    
    var_ozone_txt = ['O$_{3}$30','O$_{3}$50',
                     'TCO',
                     'O$_{3}$30','O$_{3}$50',
                     'O$_{3}$30','O$_{3}$50',
                     'TCO','TCO','TCO']
    
    
    len_vartemp = len(var_temp)
    
    #preparation:
    len_ylab = np.shape(concatenated_df_pearson.iloc[:,0])[0]
    ylab_new = []
    for l in range(len_ylab):
        ylab_new.append(str(np.ravel(concatenated_df_pearson.iloc[:,0])[l]))
    
    
    xlab = []
    for k in range(1,len_ylab+1):
        xlab.append(concatenated_df_pearson.columns[k])
        
        
    n = np.size(xlab)+1 # drop the index column and choose just every variable columns e.g. above
    val_pearson = []
    val_pvalue = []
    
    for h in range(len_vartemp):
        for i in range(0,len_ylab):
            for j in range(1,len_ylab+1):
                val_pearson.append(concatenated_df_pearson.iloc[i,j+(h*n)])
                val_pvalue.append(concatenated_df_pvalue.iloc[i,j+(h*n)])            
    
    val_pearson_new = np.reshape(val_pearson,(len_vartemp,len_ylab,len_ylab))
    
    val_pvalue_new = np.reshape(val_pvalue,(len_vartemp,len_ylab,len_ylab))
    
        
    #final_data_pearson = np.zeros((9,91,91))
    #final_data_pvalue = np.zeros((9,91,91))
    #
    #
    #for m in range(9):
    #    for n in range(91):
    #        for o in range(91):
    #            final_data_pearson[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[2][n,o]
    #            final_data_pvalue[m,n,o] = masking(val_pearson_new[m],val_pvalue_new[m])[3][n,o]
    #            print(''+str(m)+'-'+str(n)+'-'+str(o)+'')
    
    
    
    final_data0 = masking(val_pearson_new[0],val_pvalue_new[0])
    final_data1 = masking(val_pearson_new[1],val_pvalue_new[1])
    final_data2 = masking(val_pearson_new[2],val_pvalue_new[2])
    final_data3 = masking(val_pearson_new[3],val_pvalue_new[3])
    final_data4 = masking(val_pearson_new[4],val_pvalue_new[4])
    final_data5 = masking(val_pearson_new[5],val_pvalue_new[5])
    final_data6 = masking(val_pearson_new[6],val_pvalue_new[6])
    final_data7 = masking(val_pearson_new[7],val_pvalue_new[7])
    final_data8 = masking(val_pearson_new[8],val_pvalue_new[8])
    final_data9 = masking(val_pearson_new[9],val_pvalue_new[9])
    
    #plots
    fig = plt.figure(1, figsize=(20,30))  
    fig.add_subplot(321)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)

    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=20,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]

    cma = plt.get_cmap(newcmp)
    
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data1[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=20,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
        
    
    plt.savefig(r''+str(path_out)+'quad6_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    ax.set_xlabel('start date', fontsize = 18)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 18)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=20,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 18)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data7[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(c) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=20,color = 'k',bbox= props1) 
     


    fig.add_subplot(224)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=18, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 16)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 16)
    
    ax.set_xticklabels([])    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data9[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data9[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(50,25,r'(d) '+str(var_temp_txt[9])+' vs. '+str(var_ozone_txt[9])+'',fontsize=20,color = 'k',bbox= props1) 

    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=18)
       
    
    
    plt.savefig(r''+str(path_out)+'quad4_final_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
