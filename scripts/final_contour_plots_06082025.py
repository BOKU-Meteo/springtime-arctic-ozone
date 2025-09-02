# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 00:56:39 2025

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

model = 'EL_ERA5' #ERA5 or MERRA2 or EL_ERA5
print('model: '+str(model)+'')

path_in = 'C:/...' #location, where your data is stored
path_out = 'C:/...' #location of your output folder

     
if model == 'EL_ERA5':
    path = path_in
    all_files_pearson = glob.glob(os.path.join(path, 'new_'+str(k_lev)+'_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'new_'+str(k_lev)+'_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)    
    
    var_temp_txt = ['$\mathregular{A_{PSC}}$30','$\mathregular{A_{PSC}}$50',
                    '$\mathregular{T_{STRAT}}$',
                    '$\mathregular{O_{3}}$30','$\mathregular{O_{3}}$50',
                    'TCO',
                    'T30','T50',
                    '$\mathregular{V_{PSC}}$']
    
    
    var_ozone_txt = ['$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$']
    
    
    len_vartemp = len(var_temp_txt)
    
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
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
     
    ax.contourf(np.flipud(final_data7[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'EL-$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=26,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)

    ax.contourf(np.flipud(final_data1[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(6.532345733622215, 52.15119013028058, '0.0'),
#     Text(28.09715231892077, 66.1405183012166, '0.1'),
#     Text(38.0, 74.06174266515951, '0.2'),
#     Text(19.0, 43.59085275190695, '0.3'),
#     Text(55.0, 88.14509491854923, '0.4'),
#     Text(3.1653638088362968, 11.0, '0.5'),
#     Text(53.26502011249001, 73.0, '0.6'),
#     Text(40.64712613217037, 44.48518186537534, '0.7')]
    
    self_man =  [(5, 55),
                 (27, 68),
                 (38.0, 74.06174266515951),
                 (19.0, 43.59085275190695),
                 (50, 82),
                 (6, 16.0),
                 (53.26502011249001, 73.0),
                 (45, 50.0)]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)
        
    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(66.82618360813902, 79.0, '-0.3'), 
#     Text(59.336812562027006, 75.0, '-0.2'), 
#     Text(49.64599407796766, 72.0, '-0.1'), 
#     Text(41.99999999999997, 89.04552722024007, '0.0'), 
#     Text(11.999999999999972, 60.67736083715161, '0.1')]
    


    self_man =  [(66.82618360813902, 79.0),
                 (59.336812562027006, 75.0),
                 (49.64599407796766, 72.0),
                 (19, 43),
                 (11.999999999999972, 60.67736083715161)]
          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)    
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(30.000000000000007, 76.64230708233474, '0.3'), 
#     Text(10.395509607059708, 49.000000000000014, '0.4'), 
#     Text(42.0, 75.38515724337543, '0.5'), 
#     Text(22.0, 45.237569120807535, '0.6'), 
#     Text(3.0, 15.397145770748313, '0.7'), 
#     Text(61.632293461995374, 71.0, '0.8')]

    self_man = [(30.000000000000007, 76.64230708233474),
                (10.395509607059708, 49.000000000000014),
                (42.0, 75.38515724337543),
                (22.0, 45.237569120807535),
                (10, 19),
                (61.632293461995374, 71.0)]    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=26,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')

    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
        
    
    plt.savefig(r''+str(path_out)+'final_quad6_diff_lev_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(8.309169169617086, 58.99999999999997, '0.4'),
#     Text(41.539787672392976, 80.00000000000003, '0.5'),
#     Text(30.097558249073494, 57.99999999999997, '0.6'),
#     Text(20.000000000000007, 35.56611945550307, '0.7'),
#     Text(10.167242661539625, 13.0, '0.8'),
#     Text(68.42489253127275, 78.0, '0.9')]
    
    self_man = [(8.309169169617086, 58.99999999999997),
                (41.539787672392976, 80.00000000000003),
                (30.097558249073494, 57.99999999999997),
                (20.000000000000007, 35.56611945550307),
                (16, 23),
                (68.42489253127275, 78.0)]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'EL-$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(6.805742691377702, 61.0, '0.0'),
#     Text(35.96212103797893, 79.0, '0.1'),
#     Text(24.0615039385586, 56.0, '0.2'),
#     Text(12.0, 34.598427297399866, '0.3'),
#     Text(49.410814591059264, 78.0, '0.4'),
#     Text(39.0, 53.77263569632049, '0.5'),
#     Text(64.251120340347, 90.0, '0.6'),
#     Text(61.86671326690562, 78.0, '0.7'),
#     Text(65.99999999999999, 77.6205922651395, '0.8'),
#     Text(72.0, 78.27215813687052, '0.9')]
    
    self_man =  [(6.805742691377702, 61.0),
                 (35.96212103797893, 79.0),
                 (24.0615039385586, 56.0),
                 (12.0, 34.598427297399866),
                 (49.410814591059264, 78.0),
                 (39.0, 53.77263569632049),
                 (57, 80),
                 (50, 60),
                 (65.99999999999999, 77.6205922651395),
                 (72.0, 78.27215813687052)]
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 30)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(13.0, 80.46723784437725, '0.1'),
#     Text(4.9999999999999964, 57.835488916066325, '0.2'),
#     Text(29.923989680384665, 78.0, '0.3'),
#     Text(1.0, 36.3099354303349, '0.4'),
#     Text(25.363316903721028, 57.0, '0.5'),
#     Text(46.551950129788096, 79.0, '0.6'),
#     Text(20.11179772464145, 32.0, '0.7'),
#     Text(43.968899582721264, 55.000000000000014, '0.8'),
#     Text(67.0, 78.0768814436553, '0.9')]

    self_man = [(13.0, 80.46723784437725),
                (4.9999999999999964, 57.835488916066325),
                (32, 80),
                (2, 40),
                (25.363316903721028, 57.0),
                (46.551950129788096, 79.0),
                (20.11179772464145, 32.0),
                (43.968899582721264, 60.000000000000014),
                (67.0, 81.0768814436553)]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=26,color = 'k',bbox= props1) 
     
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
       
    
    
    plt.savefig(r''+str(path_out)+'final_quad4_diff_lev_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 




elif model == 'MERRA2':
    path = path_in
 
    all_files_pearson = glob.glob(os.path.join(path, 'final_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'final_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)    
    
    var_temp_txt = ['$\mathregular{A_{PSC}}$30','$\mathregular{A_{PSC}}$50',
                    '$\mathregular{T_{STRAT}}$',
                    '$\mathregular{O_{3}}$30','$\mathregular{O_{3}}$50',
                    'T30','T50',
                    'TCO','$\mathregular{V_{PSC}}$','EHF']

    
    var_ozone_txt = ['$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$','$\mathregular{TCO_{M}}$','$\mathregular{TCO_{M}}$']
    
    
    len_vartemp = len(var_temp_txt)
    
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
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'$\bf{MERRA2}$' '\n' '(a) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=26,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


#    [Text(64.90710471431314, 78.0, '-0.7'),
#     Text(41.14952438478884, 42.999999999999986, '-0.6'),
#     Text(12.474597085546804, 32.0, '-0.5'),
#     Text(37.92948556775185, 76.99999999999999, '-0.4'),
#     Text(12.07161877788895, 66.99999999999999, '-0.3')]
    
    self_man = [(64.90710471431314, 78.0),
                (41.14952438478884, 55),
                (12.474597085546804, 32.0),
                (37.92948556775185, 76.99999999999999),
                (12.07161877788895, 66.99999999999999)]
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data1[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(23.0, 74.66801947101108, '0.0'),
#     Text(4.731646737400464, 47.0, '0.1'),
#     Text(36.596749774146296, 74.0, '0.2'),
#     Text(41.0, 74.2311798263751, '0.3'),
#     Text(22.0, 43.28435012334792, '0.4'),
#     Text(1.0, 14.385339056085694, '0.5'),
#     Text(30.999999999999993, 42.062622929702044, '0.6'),
#     Text(60.97594309812139, 75.0, '0.7')]


    self_man = [(23.0, 74.66801947101108),
                (4.731646737400464, 47.0),
                (30, 69),
                (45, 77),
                (22.0, 43.28435012334792),
                (6, 19),
                (30.999999999999993, 42.062622929702044),
                (60.97594309812139, 75.0)]

           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(66.78968864598178, 80.00000000000003, '-0.4'), 
#     Text(59.99999999999997, 75.75821569728578, '-0.3'), 
#     Text(45.99999999999997, 68.65385449226292, '-0.2'), 
#     Text(49.17430485593346, 89.00000000000003, '-0.1')]


    self_man = [(66.78968864598178, 80.00000000000003),
                (59.99999999999997, 75.75821569728578),
                (45.99999999999997, 68.65385449226292),
                (30, 60)]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(26.350408889136617, 78.0, '0.3'),
#     Text(8.0, 51.25955765720926, '0.4'),
#     Text(43.95300953007858, 75.99999999999999, '0.5'),
#     Text(23.14381265399527, 47.0, '0.6'),
#     Text(3.0, 19.340482718157215, '0.7'),
#     Text(29.845861545516193, 33.0, '0.8')]


    self_man = [(26.350408889136617, 78.0),
                (8.0, 51.25955765720926),
                (43.95300953007858, 75.99999999999999),
                (23.14381265399527, 47.0),
                (7, 21),
                (35, 38)]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=26,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    #cbar.ax.invert_xaxis()
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
        
    
    plt.savefig(r''+str(path_out)+'final_quad6_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(9.73669623853236, 79.0, '0.3'), 
#     Text(30.000000000000007, 85.85189228536976, '0.4'), 
#     Text(4.920843291874753, 46.0, '0.5'), 
#     Text(36.95290810816049, 75.0, '0.6'), 
#     Text(53.0, 85.70278367241701, '0.7'), 
#     Text(29.843359670019524, 43.000000000000014, '0.8'), 
#     Text(61.999999999999986, 73.91723663212396, '0.9')]


    self_man = [(9.73669623853236, 79.0),
                (25, 75),
                (4.920843291874753, 46.0),
                (36.95290810816049, 75.0),
                (50, 80),
                (29.843359670019524, 43.000000000000014),
                (61.999999999999986, 73.91723663212396)]

           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'$\bf{MERRA2}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(17.0, 75.97176896297844, '0.1'),
#     Text(2.0, 45.58818977920046, '0.2'),
#     Text(33.343183151562144, 74.0, '0.3'),
#     Text(39.0, 74.36521181272329, '0.4'),
#     Text(21.0, 42.1804238477511, '0.5'),
#     Text(3.0, 11.104553716920499, '0.6'),
#     Text(60.233639298332136, 82.0, '0.7'),
#     Text(60.30670352926829, 73.0, '0.8'),
#     Text(69.08100543119382, 75.99999999999999, '0.9')]


    self_man = [(17.0, 75.97176896297844),
                (8, 55),
                (28, 68),
                (42, 77),
                (21.0, 42.1804238477511),
                (9, 15),
                (60.233639298332136, 82.0),
                (60.30670352926829, 73.0),
                (69.08100543119382, 75.99999999999999)]
          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)

    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 30)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(2.3803805284889563, 82.00000000000003, '0.3'), 
#     Text(18.000000000000007, 76.22736211899286, '0.4'), 
#     Text(32.0, 75.73622068441506, '0.5'), 
#     Text(13.999999999999993, 46.96181931572486, '0.6'), 
#     Text(46.38650862304948, 74.00000000000003, '0.7'), 
#     Text(30.927550088640025, 43.00000000000001, '0.8'), 
#     Text(62.06648079083631, 73.00000000000003, '0.9')]


    self_man = [(5, 84),
                (18.000000000000007, 76.22736211899286),
                (32.0, 75.73622068441506),
                (13.999999999999993, 46.96181931572486),
                (46.38650862304948, 74.00000000000003),
                (30.927550088640025, 43.00000000000001),
                (62.06648079083631, 73.00000000000003)]
      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data7[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=26,color = 'k',bbox= props1) 
     


    fig.add_subplot(224)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(15.899593204341329, 85.00000000000003, '0.1'), 
#     Text(77.60924501255008, 83.00000000000003, '0.2'), 
#     Text(25.99999999999997, 76.08603580024331, '0.2'), 
#     Text(74.47765506738551, 81.00000000000003, '0.3'), 
#     Text(0.9999999999999716, 52.71940381611426, '0.3'), 
#     Text(66.9929821483137, 71.00000000000003, '0.4'), 
#     Text(33.99999999999997, 76.62617052516637, '0.4'), 
#     Text(23.127373341853428, 24.000000000000007, '0.5'), 
#     Text(41.99999999999997, 47.61030144839476, '0.5'), 
#     Text(10.999999999999972, 16.390343072316973, '0.6')]

    self_man = [(15.899593204341329, 85.00000000000003),
                (77.60924501255008, 85.00000000000003),
                (25.99999999999997, 76.08603580024331),
                (72, 75),
                (6, 60),
                (66.9929821483137, 71.00000000000003),
                (33.99999999999997, 76.62617052516637),
                (23.127373341853428, 24.000000000000007),
                (41.99999999999997, 47.61030144839476),
                (10.999999999999972, 16.390343072316973)]      
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    #print(clabels[:])
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data9[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data9[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(d) '+str(var_temp_txt[9])+' vs. '+str(var_ozone_txt[9])+'',fontsize=26,color = 'k',bbox= props1) 

    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
       
    
    
    plt.savefig(r''+str(path_out)+'final_quad4_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 




elif model == 'ERA5':
    path = path_in
 
    all_files_pearson = glob.glob(os.path.join(path, 'final_'+str(model)+'_Pearson_*.csv'))
    all_files_pvalue = glob.glob(os.path.join(path, 'final_'+str(model)+'_pvalue_*.csv'))
    
    df_from_each_file_pearson = (pd.read_csv(f) for f in all_files_pearson)
    concatenated_df_pearson = pd.concat(df_from_each_file_pearson, ignore_index=False, axis = 1)

    df_from_each_file_pvalue = (pd.read_csv(f) for f in all_files_pvalue)
    concatenated_df_pvalue = pd.concat(df_from_each_file_pvalue, ignore_index=False, axis = 1)                  
                                
    var_temp_txt = ['$\mathregular{A_{PSC}}$30','$\mathregular{A_{PSC}}$50',
                    '$\mathregular{T_{STRAT}}$',
                    '$\mathregular{O_{3}}$30','$\mathregular{O_{3}}$50',
                    'T30','T50',
                    'TCO','$\mathregular{V_{PSC}}$','EHF']
    
    var_ozone_txt = ['$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{O_{3}}$$\mathregular{30_{M}}$','$\mathregular{O_{3}}$$\mathregular{50_{M}}$',
                     '$\mathregular{TCO_{M}}$','$\mathregular{TCO_{M}}$','$\mathregular{TCO_{M}}$']
    
    
    len_vartemp = len(var_temp_txt)
    
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
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data6[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data6[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data6[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[6])+' vs. '+str(var_ozone_txt[6])+'',fontsize=26,color = 'k',bbox= props1) 
        



    fig.add_subplot(322)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


#    [Text(64.90710471431314, 78.0, '-0.7'),
#     Text(41.14952438478884, 42.999999999999986, '-0.6'),
#     Text(12.474597085546804, 32.0, '-0.5'),
#     Text(37.92948556775185, 76.99999999999999, '-0.4'),
#     Text(12.07161877788895, 66.99999999999999, '-0.3')]
    
#    self_man = [(64.90710471431314, 78.0),
#                (41.14952438478884, 55),
#                (12.474597085546804, 32.0),
#                (37.92948556775185, 76.99999999999999),
#                (12.07161877788895, 66.99999999999999)]
#    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_yticklabels([])

    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data1[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data1[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data1[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
        
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[1])+' vs. '+str(var_ozone_txt[1])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(323)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


#    [Text(0.0, 47.44780700372934, '0.0'),
#     Text(33.43321952061355, 74.0, '0.1'),
#     Text(13.777046114662724, 45.0, '0.2'),
#     Text(42.2113646256881, 74.0, '0.3'),
#     Text(46.48849065409193, 74.0, '0.4'),
#     Text(3.2636141324295913, 13.0, '0.5'),
#     Text(32.99999999999999, 41.796898056810605, '0.6'),
#     Text(40.86497152463315, 42.0, '0.7')]

    self_man = [(8, 60),
                (33.43321952061355, 74.0),
                (13.777046114662724, 45.0),
                (42.2113646256881, 74.0),
                (50, 80),
                (7, 18),
                (32.99999999999999, 41.796898056810605),
                (60, 64)]
#               
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data5[2], cmap = cma, levels=clevels1)

    cs = plt.contour(xlab,np.flipud(ylab_new), final_data5[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data5[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[5])+' vs. '+str(var_ozone_txt[5])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(324)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data0[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data0[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data0[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
          
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(d) '+str(var_temp_txt[0])+' vs. '+str(var_ozone_txt[0])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(325)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    
#    [Text(12.45215048998163, 80.0, '0.3'),
#     Text(34.0, 76.75701514151474, '0.4'),
#     Text(13.0, 49.55796155854543, '0.5'),
#     Text(55.0, 83.32741547455917, '0.6'),
#     Text(2.747807611032936, 19.999999999999996, '0.7'),
#     Text(32.386083724950865, 38.0, '0.8')]

    self_man = [(12.45215048998163, 80.),
                (34.0, 76.75701514151474),
                (13.0, 49.55796155854543),
                (55.0, 83.32741547455917),
                (5, 25),
                (32.386083724950865, 38.0)]

           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data2[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data2[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data2[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(e) '+str(var_temp_txt[2])+' vs. '+str(var_ozone_txt[2])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    
    fig.add_subplot(326)
    ax = plt.gca()
    clevels1 = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    

#    [Text(69.88701198890831, 79.00000000000001, '-0.7'), 
#     Text(55.21909919787504, 71.00000000000001, '-0.6'), 
#     Text(25.709674165649886, 39.000000000000014, '-0.5'), 
#     Text(47.99999999999997, 89.91614230168945, '-0.4'), 
#     Text(16.430112106593157, 66.00000000000001, '-0.3'), 
#     Text(7.447593394604667, 79.99999999999999, '-0.2')]

    self_man = [(69.88701198890831, 79.00000000000001),
                (55.21909919787507, 71.0),
                (25.709674165649915, 39.0),
                (40,80),
                (16.430112106593192, 66.0),
                (7.447593394604667, 79.99999999999999)]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp)
     
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data8[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data8[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data8[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(f) '+str(var_temp_txt[8])+' vs. '+str(var_ozone_txt[8])+'',fontsize=26,color = 'k',bbox= props1) 
    
    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    #cbar.ax.invert_xaxis()
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
        
    
    plt.savefig(r''+str(path_out)+'final_quad6_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 
    
    
    
    
    
    fig = plt.figure(1, figsize=(20,20))  
    fig.add_subplot(221)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
           
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    ax.set_xlabel('start date', fontsize = 30)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data4[2], cmap = cma, levels=clevels1)
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data4[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data4[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
           
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'$\bf{ERA5}$' '\n' '(a) '+str(var_temp_txt[4])+' vs. '+str(var_ozone_txt[4])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    fig.add_subplot(222)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(22.0, 74.28834368445561, '0.0'),
#     Text(5.33768474903032, 44.0, '0.1'),
#     Text(12.0, 43.21995326344448, '0.2'),
#     Text(41.0, 73.79426317090507, '0.3'),
#     Text(0.0, 11.98222867144348, '0.4'),
#     Text(50.0, 73.19156386992545, '0.5'),
#     Text(33.80480014937527, 40.0, '0.6'),
#     Text(68.0, 88.7872813749257, '0.7'),
#     Text(63.584114496642314, 73.0, '0.8')]

    self_man = [(22.0, 74.28834368445561),
                (5.33768474903032, 44.0),
                (12.0, 43.21995326344448),
                (41.0, 73.79426317090507),
                (15, 30),
                (50.0, 73.19156386992545),
                (33.80480014937527, 40.0),
                (65, 85),
                (63.584114496642314, 73.0)]

          
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_xlabel('start date', fontsize = 30)
    
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data3[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data3[2], colors='black', linewidths=1 ,levels= clevels1)
    
    
    ax.contourf(np.flipud(final_data3[3]),colors='white') 
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(b) '+str(var_temp_txt[3])+' vs. '+str(var_ozone_txt[3])+'',fontsize=26,color = 'k',bbox= props1) 
        
    
    
    
    fig.add_subplot(223)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#    [Text(8.963468529910774, 79.0, '0.2'),
#     Text(14.878379856588836, 75.0, '0.3'),
#     Text(0.0, 43.29074836626286, '0.4'),
#     Text(32.0, 74.37304338479188, '0.5'),
#     Text(15.438853322543192, 44.0, '0.6'),
#     Text(57.373939580210305, 90.0, '0.7'),
#     Text(9.0, 11.58473968006286, '0.8'),
#     Text(63.131463716963566, 73.0, '0.9')]

    self_man = [(8.963468529910774, 79.0),
                (16, 77.0),
                (8, 50),
                (32.0, 74.37304338479188),
                (15.438853322543192, 44.0),
                (50, 75),
                (40, 40),
                (63.131463716963566, 73.0)]

      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    ax.set_ylabel('number of days', fontsize = 30)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data7[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data7[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data7[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(c) '+str(var_temp_txt[7])+' vs. '+str(var_ozone_txt[7])+'',fontsize=26,color = 'k',bbox= props1) 
     


    fig.add_subplot(224)
    ax = plt.gca()
    
    clevels1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
#    [Text(76.74946894326459, 83.00000000000003, '0.2'), 
#     Text(13.675815338965663, 83.00000000000003, '0.2'),
#     Text(72.6155100242247, 81.00000000000003, '0.3'),
#     Text(28.439019511571047, 77.00000000000003, '0.3'),
#     Text(59.99999999999997, 75.07107684892995, '0.4'),
#     Text(3.904426179999234, 52.00000000000002, '0.4'), 
#     Text(43.99999999999997, 49.57754372429712, '0.5'), 
#     Text(15.436336870236403, 26.000000000000007, '0.6'), 
#     Text(10.72352535065005, 33.00000000000001, '0.7')]

    self_man = [(76.74946894326459, 85.00000000000003),
                (13.675815338965663, 83.00000000000003),
                (72.6155100242247, 75.00000000000003),
                (28.439019511571047, 77.00000000000003),
                (59.99999999999997, 75.07107684892995),
                (3.904426179999234, 52.00000000000002),
                (43.99999999999997, 49.57754372429712),
                (15.436336870236403, 26.000000000000007),
                (10.72352535065005, 33.00000000000001)]


      
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=0.6, levels= clevels1)
    cs.collections[10].set_linewidth(2) 
    
    #clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f')
    clabels = plt.clabel(cs, clevels1, rightside_up=True, colors = 'k', fontsize=30, fmt='%1.1f', manual = self_man)
    #clabels[:]
    #print(clabels[:])
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1)) for txt in clabels]
    [txt.set_fontweight('bold') for txt in clabels]
    
    cma = plt.get_cmap(newcmp1)
    
    ax.xaxis.tick_top()
    plt.yticks(np.arange(91)[::10], labels=np.flipud(ylab_new).astype(int)[::10], fontsize = 28)
    plt.xticks(np.arange(91)[::10], labels=xlab[::10], rotation = 90, fontsize = 28)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    contour_filled = plt.contourf(xlab,np.flipud(ylab_new), final_data9[2], cmap = cma, levels=clevels1)
    
    
    cs = plt.contour(xlab,np.flipud(ylab_new), final_data9[2], colors='black', linewidths=1 ,levels= clevels1)
    
    ax.contourf(np.flipud(final_data9[3]),colors='white')
    
    plt.plot([0,90], [0,90], 'k-', linewidth=1.4)
    
    props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey')
    plt.text(35,20,r'(d) '+str(var_temp_txt[9])+' vs. '+str(var_ozone_txt[9])+'',fontsize=26,color = 'k',bbox= props1) 

    
    fig.subplots_adjust(bottom=0.125, hspace=0.050, wspace=0.050)
    cb_ax = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    cbar = fig.colorbar(contour_filled, cax=cb_ax,orientation='horizontal')
    
    cbar.set_ticks(clevels1)
    cbar.set_ticklabels(clevels1)
    cbar.ax.tick_params(labelsize=20)
       
    
    
    plt.savefig(r''+str(path_out)+'final_quad4_'+str(model)+'_contour_all_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300) 
    plt.close() 