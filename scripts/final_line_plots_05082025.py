# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 22:06:31 2025

@author: Jevare
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter  
from datetime import datetime
import glob
import os

#---------------------------------

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.0f' % (x * 1e-6)

#---------------------------------
len_mon = [31,30,31,31,30,31,30,31,31,29,31,30]
num_start = 1
month_starts = [num_start,
                num_start+len_mon[0],
                num_start+len_mon[0]+len_mon[1],
                num_start+len_mon[0]+len_mon[1]+len_mon[2],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6]+len_mon[7],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6]+len_mon[7]+len_mon[8],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6]+len_mon[7]+len_mon[8]+len_mon[9],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6]+len_mon[7]+len_mon[8]+len_mon[9]+len_mon[10],
                num_start+len_mon[0]+len_mon[1]+len_mon[2]+len_mon[3]+len_mon[4]+len_mon[5]+len_mon[6]+len_mon[7]+len_mon[8]+len_mon[9]+len_mon[10]+len_mon[11]]
month_names = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul'] 


len_years = np.linspace(1980,2024,45).astype(int)
len_years_red = np.linspace(1980,2023,44).astype(int)

#--------------------------------
model = 'EL_ERA5' #ERA5 or MERRA2 or EL_ERA5
print('model: '+str(model)+'')

path_in = 'C:/...' #location, where your data is stored
path_out = 'C:/...' #location of your output folder

if model == 'EL_ERA5':
    path = ''+str(path_in)+''

    all_files = glob.glob(os.path.join(path, 'new_530_final_'+str(model)+'_*.csv'))
    
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=False, axis = 1)


    var_txt = ['Temperature_50hPa','Temperature_30hPa','O3_50hPa','O3_30hPa','TO3','Apsc_50hPa','Apsc_30hPa','Vpsc','m_strat_T', 'Day', 'Month', 'Year']
    n = np.size(var_txt)+1 # drop the index column and choose just every variable columns e.g. above
    val_new = []
    val_rev = []
    
    
    cal_jul = 31+28+31+30+31+30 # Jan til Jun in a non leap year
    pos = 59 # position of 29th of February
    dpy = np.shape(concatenated_df)[0]-1 # 365 days
    
    for i in range(len(len_years)):
        val = concatenated_df.iloc[:, [1+(n*i),2+(n*i),3+(n*i),4+(n*i),5+(n*i),6+(n*i),7+(n*i),8+(n*i),9+(n*i),10+(n*i),11+(n*i),12+(n*i)]]
        if ((val['Month'] == 2) & (val['Day'] == 29))[pos] == True:
            val = val.drop(pos)
            val.reset_index(inplace=True, drop=True) # index is wrong after drop, set reindex
            val_new.append(val)
            #print('!!! '+str(int(val['Year'][0]))+' is a leap year')
        else:
            val = val.drop(dpy)
            val.reset_index(inplace=True, drop=True)
            val_new.append(val)       
            #print(''+str(int(val['Year'][0]))+' is no leap year')
    
    for i in range(len(len_years)):
        if i == len(len_years_red):
            break
        else:
            valt = np.concatenate((val_new[i].loc[cal_jul:][0:],val_new[i+1].loc[0:cal_jul-1][:]))
            valt = pd.DataFrame(valt)
            valt.columns = var_txt
            val_rev.append(valt)
    
    #-----------------------
    per_model = [25,50,75]
    
    var_temp_30hPa = 'Temperature_30hPa'
    var_temp_50hPa = 'Temperature_50hPa'
    var_ozone_30hPa = 'O3_30hPa'
    var_ozone_50hPa = 'O3_50hPa'
    var_to3 = 'TO3'
    var_vpsc = 'Vpsc'
    var_m_strat_T = 'm_strat_T'
    
    var_ozone_all = 'O3'
    var_temp_all = 'T'
    
    len_days = np.linspace(0,len(val_rev[0])+1,len(val_rev[0]))
    
    
    data_all_temp_30hPa = np.zeros((44,365))
    data_all_temp_50hPa = np.zeros((44,365))
    data_all_ozone_30hPa = np.zeros((44,365))
    data_all_ozone_50hPa = np.zeros((44,365))
    data_all_to3 = np.zeros((44,365))
    data_all_vpsc = np.zeros((44,365))
    data_all_m_strat_T = np.zeros((44,365))
    
    for ll in range(44):
        for mm in range(365):
            data_all_temp_30hPa[ll,mm] = float(val_rev[ll][var_temp_30hPa][mm])
            data_all_temp_50hPa[ll,mm] = float(val_rev[ll][var_temp_50hPa][mm])
            data_all_ozone_30hPa[ll,mm] = float(val_rev[ll][var_ozone_30hPa][mm])
            data_all_ozone_50hPa[ll,mm] = float(val_rev[ll][var_ozone_50hPa][mm])
            data_all_to3[ll,mm] = float(val_rev[ll][var_to3][mm])
            data_all_vpsc[ll,mm] = float(val_rev[ll][var_vpsc][mm])
            data_all_m_strat_T[ll,mm] = float(val_rev[ll][var_m_strat_T][mm])        
             
             
    
    new_data_plot = [data_all_temp_50hPa,data_all_temp_50hPa,
                     data_all_temp_30hPa,data_all_temp_30hPa,
                     data_all_ozone_50hPa,data_all_ozone_50hPa,
                     data_all_ozone_30hPa,data_all_ozone_30hPa,
                     data_all_m_strat_T,data_all_m_strat_T,
                     data_all_vpsc,data_all_vpsc,
                     data_all_to3,data_all_to3]
    
    
    len_data_all = np.shape(data_all_temp_30hPa)[0] #44
    len_year_days = np.shape(data_all_temp_30hPa)[1] #365
    len_new_data_plot = len(new_data_plot) # 16
    len_per_model = len(per_model) #3
    
    per_all_plot = np.zeros((len_new_data_plot,len_year_days,len_per_model))
    max_all_plot = np.zeros((len_new_data_plot,len_year_days))
    min_all_plot = np.zeros((len_new_data_plot,len_year_days))
    mean_all_plot = np.zeros((len_new_data_plot,len_year_days))
    median_all_plot = np.zeros((len_new_data_plot,len_year_days))
    
    for nn in range(len_new_data_plot):
        for oo in range(len_year_days):
            temp_data = []
            for qq in range(len_data_all):
                temp_data.append(new_data_plot[nn][qq][oo])
                #print('nn: '+str(nn)+ ',qq: '+str(qq)+ ',oo: '+str(oo)+'')
                #print(''+str(temp_data[0])+'')
                for pp in range(len_per_model):            
                    per_all_plot[nn,oo,pp] = np.nanpercentile(temp_data,per_model[pp])
            max_all_plot[nn,oo] = np.nanmax(temp_data)       
            min_all_plot[nn,oo] = np.nanmin(temp_data)       
            mean_all_plot[nn,oo] = np.nanmean(temp_data)       
            median_all_plot[nn,oo] = np.nanmedian(temp_data)       
    
    
    #---------------------------------------
        
    len_years_red = np.linspace(1980,2023,44).astype(int)
    
    n_yea_early = [1980,1999,2004,2015]
    len_yea_early = len(n_yea_early)
    
    n_yea_late = [1996,2010,2019,2021]
    len_yea_late = len(n_yea_late)
       
    col_late = ['limegreen','darkcyan','blue','purple','black','grey','grey','grey']
    col_early_red = ['darkgoldenrod','darkorange','red','brown','black','grey','grey','grey']
    
    leg_var_both = ['1980/81','1999/00','2004/05','2015/16','1996/97','2010/11','2019/20','2021/22',
                    'Mean','Max','Min','Q25-Q75']
    col_both = ['darkgoldenrod','darkorange','red','brown','limegreen','darkcyan','blue','purple',
                'black','grey','grey','grey']       
    
    #--------------------------------------------------------------------------

    #plot configurations
    
    sel_start_xlim = [62,62,62,62,62,62,62,62,62,62,62,62,62,62]
    sel_end_xlim = [337,337,337,337,337,337,337,337,337,337,337,337,337,337]
    
    sel_start_ylim = [185,185,185,185,2*10**-6,2*10**-6,2*10**-6,2*10**-6,185,185,0,0,220,220]
    sel_end_ylim = [245,245,245,245,7.6*10**-6,7.6*10**-6,1.05*10**-5,1.05*10**-5,245,245,3.5*10**7,3.5*10**7,530,530]
    
    sel_txt = [r'EL-$\bf{ERA5}$' '\n' '(a) T50',r'(b) T50',r'(c) T30',r'(d) T30',
               r'(e) O$_{3}$50',r'(f) O$_{3}$50',r'(g) O$_{3}$30',r'(h) O$_{3}$30',
               r'EL-$\bf{ERA5}$' '\n' '(a) $\mathregular{T_{STRAT}}$',r'(b) $\mathregular{T_{STRAT}}$',
               r'(c) $\mathregular{V_{PSC}}$',r'(d) $\mathregular{V_{PSC}}$',
               r'(d) TCO',r'(e) TCO']
    
    sel_col = [col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late]
    
    sel_start_loc = [70,70,70,70,70,70,70,70,70,70,70,70,70,70]
    
    sel_end_loc = [236.5,240,
                   240,240,
                   7*10**-6,7*10**-6,
                   9.7*10**-6,9.7*10**-6,
                   235.5,238.5,
                   3.2*10**7,3.2*10**7,
                   500,500]

    sel_n_yea = [n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,                
                 n_yea_early,n_yea_late]
    
    sel_len_yea = [len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late]
       
    sel_multi = [10,10,10,10,
                 10**-6,10**-6,
                 10**-6,10**-6,
                 10,10,
                 0.5*10**7,0.5*10**7,
                 40,40]

    n_legend = 26
    n_txt = 28
    n_ylabel = 26
    n_xlabel = 26
    n_xyticks = 24 
    
    #--------------------------------------Plots    
    
    #1 DU = 2.1415 x 10-5 kg[O3]/m2

    lns_mean = []
    lns_max = []
    lns_min = []
    lns_early = []
    lns_late = []
    lns_fill = []
    
    fig, axs = plt.subplots(4,2, figsize=(20, 30), facecolor='w', edgecolor='k')
           
    axs = axs.ravel()
           
    for i_plots in range(10):
    
        if i_plots == 0:    
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            lns_mean.extend(axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4))
            lns_max.extend(axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4))
            lns_min.extend(axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4))
            
            for l6 in range(len_yea_early):
                lns_early.extend(axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            lns_fill.append(axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4))       
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()   
     
        elif i_plots == 1:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                lns_late.extend(axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)      
            
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
    
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()   
    
        elif i_plots == 2:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
         
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()  
    
        elif i_plots == 3:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
    
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()  
            
        elif i_plots == 4:  
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('mg/kg',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()     
    
             
        elif i_plots == 5:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()     
    
    
        elif i_plots == 6:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('mg/kg',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()    
    
    
        elif i_plots == 7:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                    
            axs[i_plots].set_yticklabels([])
            axs[i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()    
            

    #make a new legend with all lines
    lns = lns_early+lns_late+lns_mean+lns_max+lns_min+lns_fill

    axs[0].legend(lns, leg_var_both, prop={'size': n_legend},loc='upper center', bbox_to_anchor=(1, 1.39),
                      ncol=3, fancybox=True, shadow=True)    
            
    plt.savefig(r''+str(path_out)+'final_quad8_'+str(model)+'_'+str(var_temp_all)+'_'+str(var_ozone_all)+'_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300, bbox_inches='tight')
    plt.close()  
    
    
    #----------------------------

    lns_mean = []
    lns_max = []
    lns_min = []
    lns_early = []
    lns_late = []
    lns_fill = []    
    
    fig, axs = plt.subplots(3,2, figsize=(20, 20), facecolor='w', edgecolor='k')
           
    axs = axs.ravel()
           
    for i_plots in range(8,14):
    
        if i_plots == 8:    
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots-i_plots].set_xticks(month_starts)
            axs[i_plots-i_plots].set_xticklabels(month_names)
            axs[i_plots-i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            lns_mean.extend(axs[i_plots-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4))
            lns_max.extend(axs[i_plots-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4))
            lns_min.extend(axs[i_plots-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4))
            
            for l6 in range(len_yea_early):
                lns_early.extend(axs[i_plots-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            lns_fill.append(axs[i_plots-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4))
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots-i_plots].set_xticklabels([])
            plt.tight_layout()   
     
        elif i_plots == 9:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+1-i_plots].set_xticks(month_starts)
            axs[i_plots+1-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+1-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+1-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+1-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                lns_late.extend(axs[i_plots+1-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            axs[i_plots+1-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)      
            
            axs[i_plots+1-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+1-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+1-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+1-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
    
            axs[i_plots+1-i_plots].set_xticklabels([])
            axs[i_plots+1-i_plots].set_yticklabels([])
            plt.tight_layout()   
    
        elif i_plots == 10:
            formatter = FuncFormatter(millions)
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+2-i_plots].set_xticks(month_starts)
            axs[i_plots+2-i_plots].set_xticklabels(month_names)
            axs[i_plots+2-i_plots].set_ylabel(r'million km$^{3}$',fontsize=n_ylabel)
            
            axs[i_plots+2-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+2-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+2-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+2-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+2-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
         
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                
            axs[i_plots+2-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+2-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+2-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+2-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+2-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots+2-i_plots].set_xticklabels([])
            axs[i_plots+2-i_plots].yaxis.set_major_formatter(formatter) 
            
            plt.tight_layout()  
    
        elif i_plots == 11:
            formatter = FuncFormatter(millions)
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+3-i_plots].set_xticks(month_starts)
            axs[i_plots+3-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+3-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+3-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+3-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+3-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+3-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots+3-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
    
            axs[i_plots+3-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+3-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots+3-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots+3-i_plots].yaxis.set_major_formatter(formatter) 
            axs[i_plots+3-i_plots].set_xticklabels([])
            axs[i_plots+3-i_plots].set_yticklabels([])
            plt.tight_layout()  
    
    
        elif i_plots == 12:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+4-i_plots].set_xticks(month_starts)
            axs[i_plots+4-i_plots].set_xticklabels(month_names)
            axs[i_plots+4-i_plots].set_ylabel('DU',fontsize=n_ylabel)
            
            axs[i_plots+4-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+4-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+4-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+4-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+4-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots+4-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+4-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+4-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+4-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+4-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots+4-i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()    
    
    
        elif i_plots == 13:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+5-i_plots].set_xticks(month_starts)
            axs[i_plots+5-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+5-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+5-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+5-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+5-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+5-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
        
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots+5-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+5-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+5-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots+5-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+5-i_plots].set_yticklabels([])
            axs[i_plots+5-i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout() 
    
    
    #make a new legend with all lines
    lns = lns_early+lns_late+lns_mean+lns_max+lns_min+lns_fill

    axs[0].legend(lns, leg_var_both, prop={'size': n_legend},loc='upper center', bbox_to_anchor=(1, 1.42),
                      ncol=3, fancybox=True, shadow=True)        
    
    plt.savefig(r''+str(path_out)+'final_quad6_'+str(model)+'_'+str(var_m_strat_T)+'_'+str(var_vpsc)+'_'+str(var_to3)+'_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300, bbox_inches='tight')
    plt.close()  


# for ERA5 or MERRA2
else:
    path = 'C:/Users/Jevare/Desktop/data_ozonpaper/'
    #path = 'C:/Users/Jevare/Desktop/Boku/Ales/'+str(model)+'/'  
    all_files = glob.glob(os.path.join(path, 'final_'+str(model)+'_all*.csv'))
    
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=False, axis = 1)

    var_txt = ['Temperature_50hPa','Temperature_30hPa','O3_50hPa','O3_30hPa','TO3','Apsc_50hPa','Apsc_30hPa','Vpsc','vT','m_strat_T', 'Day', 'Month', 'Year']
    n = np.size(var_txt)+1 # drop the index column and choose just every variable columns e.g. above
    val_new = []
    val_rev = []
    
    
    cal_jul = 31+28+31+30+31+30 # Jan til Jun in a non leap year
    pos = 59 # position of 29th of February
    dpy = np.shape(concatenated_df)[0]-1 # 365 days
    
    for i in range(len(len_years)):
        val = concatenated_df.iloc[:, [1+(n*i),2+(n*i),3+(n*i),4+(n*i),5+(n*i),6+(n*i),7+(n*i),8+(n*i),9+(n*i),10+(n*i),11+(n*i),12+(n*i),13+(n*i)]]
        if ((val['Month'] == 2) & (val['Day'] == 29))[pos] == True:
            val = val.drop(pos)
            val.reset_index(inplace=True, drop=True) # index is wrong after drop, set reindex
            val_new.append(val)
            #print('!!! '+str(int(val['Year'][0]))+' is a leap year')
        else:
            val = val.drop(dpy)
            val.reset_index(inplace=True, drop=True)
            val_new.append(val)       
            #print(''+str(int(val['Year'][0]))+' is no leap year')
    
    for i in range(len(len_years)):
        if i == len(len_years_red):
            break
        else:
            valt = np.concatenate((val_new[i].loc[cal_jul:][0:],val_new[i+1].loc[0:cal_jul-1][:]))
            valt = pd.DataFrame(valt)
            valt.columns = var_txt
            val_rev.append(valt)
                    
    #-----------------------
    per_model = [25,50,75]

    var_temp_30hPa = 'Temperature_30hPa'
    var_temp_50hPa = 'Temperature_50hPa'
    var_ozone_30hPa = 'O3_30hPa'
    var_ozone_50hPa = 'O3_50hPa'
    var_to3 = 'TO3'
    var_vpsc = 'Vpsc'
    var_vT = 'vT'
    var_m_strat_T = 'm_strat_T'
    
    var_ozone_all = 'O3'
    var_temp_all = 'T'
    
    len_days = np.linspace(0,len(val_rev[0])+1,len(val_rev[0]))
    
    
    data_all_temp_30hPa = np.zeros((44,365))
    data_all_temp_50hPa = np.zeros((44,365))
    data_all_ozone_30hPa = np.zeros((44,365))
    data_all_ozone_50hPa = np.zeros((44,365))
    data_all_to3 = np.zeros((44,365))
    data_all_vpsc = np.zeros((44,365))
    data_all_vT = np.zeros((44,365))
    data_all_m_strat_T = np.zeros((44,365))
    
    for ll in range(44):
        for mm in range(365):
            data_all_temp_30hPa[ll,mm] = float(val_rev[ll][var_temp_30hPa][mm])
            data_all_temp_50hPa[ll,mm] = float(val_rev[ll][var_temp_50hPa][mm])
            data_all_ozone_30hPa[ll,mm] = float(val_rev[ll][var_ozone_30hPa][mm])
            data_all_ozone_50hPa[ll,mm] = float(val_rev[ll][var_ozone_50hPa][mm])
            data_all_to3[ll,mm] = float(val_rev[ll][var_to3][mm])
            data_all_vpsc[ll,mm] = float(val_rev[ll][var_vpsc][mm])
            data_all_vT[ll,mm] = float(val_rev[ll][var_vT][mm])
            data_all_m_strat_T[ll,mm] = float(val_rev[ll][var_m_strat_T][mm])        
             
             
    
    new_data_plot = [data_all_temp_50hPa,data_all_temp_50hPa,
                     data_all_temp_30hPa,data_all_temp_30hPa,
                     data_all_ozone_50hPa,data_all_ozone_50hPa,
                     data_all_ozone_30hPa,data_all_ozone_30hPa,
                     data_all_vT,data_all_vT,
                     data_all_m_strat_T,data_all_m_strat_T,
                     data_all_vpsc,data_all_vpsc,
                     data_all_to3,data_all_to3]
    
    
    len_data_all = np.shape(data_all_temp_30hPa)[0] #44
    len_year_days = np.shape(data_all_temp_30hPa)[1] #365
    len_new_data_plot = len(new_data_plot) # 16
    len_per_model = len(per_model) #3
    
    per_all_plot = np.zeros((len_new_data_plot,len_year_days,len_per_model))
    max_all_plot = np.zeros((len_new_data_plot,len_year_days))
    min_all_plot = np.zeros((len_new_data_plot,len_year_days))
    mean_all_plot = np.zeros((len_new_data_plot,len_year_days))
    median_all_plot = np.zeros((len_new_data_plot,len_year_days))
    
    for nn in range(len_new_data_plot):
        for oo in range(len_year_days):
            temp_data = []
            for qq in range(len_data_all):
                temp_data.append(new_data_plot[nn][qq][oo])
                #print('nn: '+str(nn)+ ',qq: '+str(qq)+ ',oo: '+str(oo)+'')
                #print(''+str(temp_data[0])+'')
                for pp in range(len_per_model):            
                    per_all_plot[nn,oo,pp] = np.nanpercentile(temp_data,per_model[pp])
            max_all_plot[nn,oo] = np.nanmax(temp_data)       
            min_all_plot[nn,oo] = np.nanmin(temp_data)       
            mean_all_plot[nn,oo] = np.nanmean(temp_data)       
            median_all_plot[nn,oo] = np.nanmedian(temp_data)       
    
    
    #---------------------------------------
        
    len_years_red = np.linspace(1980,2023,44).astype(int)
    
    n_yea_early = [1980,1999,2004,2015]
    len_yea_early = len(n_yea_early)
    
    n_yea_late = [1996,2010,2019,2021]
    len_yea_late = len(n_yea_late)
    
    col_late = ['limegreen','darkcyan','blue','purple','black','grey','grey','grey']
    col_early_red = ['darkgoldenrod','darkorange','red','brown','black','grey','grey','grey']
       
    leg_var_both = ['1980/81','1999/00','2004/05','2015/16','1996/97','2010/11','2019/20','2021/22',
                    'Mean','Max','Min','Q25-Q75']
    col_both = ['darkgoldenrod','darkorange','red','brown','limegreen','darkcyan','blue','purple',
                'black','grey','grey','grey']    
    
    #--------------------------------------------------------------------------
    
    #plot configurations
    
    sel_start_xlim = [62,62,62,62,62,62,62,62,62,62,62,62,62,62,62,62]
    sel_end_xlim = [337,337,337,337,337,337,337,337,337,337,337,337,337,337,337,337]
    
    sel_start_ylim = [190,190,190,190,3.5*10**-6,3.5*10**-6,4.3*10**-6,4.3*10**-6,-25,-25,195,195,0,0,255,255]
    sel_end_ylim = [237,237,237,237,7.1*10**-6,7.1*10**-6,1*10**-5,1*10**-5,70.5,70.5,235,235,8.7*10**7,8.7*10**7,500,500]
    
    sel_txt = [r'$\bf{'+str(model)+'}$' '\n' '(a) T50',r'(b) T50',r'(c) T30',r'(d) T30',
               r'(e) $\mathregular{O_{3}}$50',r'(f) $\mathregular{O_{3}}$50',r'(g) $\mathregular{O_{3}}$30',r'(h) $\mathregular{O_{3}}$30',
               r'(i) EHF',r'(j) EHF',
               r'$\bf{'+str(model)+'}$' '\n' '(a) $\mathregular{T_{STRAT}}$',r'(b) $\mathregular{T_{STRAT}}$',
               r'(c) $\mathregular{V_{PSC}}$',r'(d) $\mathregular{V_{PSC}}$',
               r'(d) TCO',r'(e) TCO']
    
    sel_col = [col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late,
               col_early_red,col_late]
    
    
    sel_start_loc = [70,70,70,70,70,70,70,70,70,70,70,70,70,70,70,70]


    sel_end_loc = [228.5,232.5,232.5,232.5,
                   6.75*10**-6,6.75*10**-6,
                   9.45*10**-6,9.45*10**-6,
                   60,60,
                   228.3,231.3,
                   7.9*10**7,7.9*10**7,
                   477,477]
     
    sel_n_yea = [n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late,                
                 n_yea_early,n_yea_late,
                 n_yea_early,n_yea_late]
    
    sel_len_yea = [len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late,
                   len_yea_early,len_yea_late]
    
    sel_multi = [10,10,10,10,
                 5*10**-7,5*10**-7,
                 10**-6,10**-6,
                 20,20,
                 10,10,
                 2*10**7,2*10**7,
                 40,40]
    
    
    n_legend = 26
    n_txt = 28
    n_ylabel = 26
    n_xlabel = 26
    n_xyticks = 24

    #--------------------------------------Plots    
    
    #1 DU = 2.1415 x 10-5 kg[O3]/m2

    lns_mean = []
    lns_max = []
    lns_min = []
    lns_early = []
    lns_late = []
    lns_fill = []
    
    fig, axs = plt.subplots(5,2, figsize=(20, 30), facecolor='w', edgecolor='k')
           
    axs = axs.ravel()
           
    for i_plots in range(10):
    
        if i_plots == 0:    
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            lns_mean.extend(axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4))
            lns_max.extend(axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4))
            lns_min.extend(axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4))
            
            for l6 in range(len_yea_early):
                lns_early.extend(axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            lns_fill.append(axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4))       
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()   
     
        elif i_plots == 1:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                lns_late.extend(axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)      
            
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
    
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()   
    
        elif i_plots == 2:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
         
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()  
    
        elif i_plots == 3:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
    
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()  
            
        elif i_plots == 4:  
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('mg/kg',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()     
    
             
        elif i_plots == 5:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()     
    
    
        elif i_plots == 6:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('mg/kg',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].set_xticklabels([])
            plt.tight_layout()    
    
    
        elif i_plots == 7:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
                
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1)         
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                    
            axs[i_plots].set_xticklabels([])
            axs[i_plots].set_yticklabels([])
            plt.tight_layout()    
            
    
    
        elif i_plots == 8:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            axs[i_plots].set_ylabel('Km/s',fontsize=n_ylabel)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()    
    
    
        elif i_plots == 9:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots].set_xticks(month_starts)
            axs[i_plots].set_xticklabels(month_names)
            
            axs[i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
        
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots].set_yticklabels([])
            axs[i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()  

    #make a new legend with all lines
    lns = lns_early+lns_late+lns_mean+lns_max+lns_min+lns_fill

    axs[0].legend(lns, leg_var_both, prop={'size': n_legend},loc='upper center', bbox_to_anchor=(1, 1.49),
                      ncol=3, fancybox=True, shadow=True)    
            
    plt.savefig(r''+str(path_out)+'final_quad8_'+str(model)+'_'+str(var_temp_all)+'_'+str(var_ozone_all)+'_'+str(var_vT)+'_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300, bbox_inches='tight')
    plt.close()  
    
    
    #----------------------------
    
    lns_mean = []
    lns_max = []
    lns_min = []
    lns_early = []
    lns_late = []
    lns_fill = []
    
    fig, axs = plt.subplots(3,2, figsize=(20, 20), facecolor='w', edgecolor='k')
           
    axs = axs.ravel()
           
    for i_plots in range(10,16):
    
        if i_plots == 10:    
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots-i_plots].set_xticks(month_starts)
            axs[i_plots-i_plots].set_xticklabels(month_names)
            axs[i_plots-i_plots].set_ylabel('K',fontsize=n_ylabel)
            
            lns_mean.extend(axs[i_plots-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4))
            lns_max.extend(axs[i_plots-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4))
            lns_min.extend(axs[i_plots-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4))
            
            for l6 in range(len_yea_early):
                lns_early.extend(axs[i_plots-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            lns_fill.append(axs[i_plots-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4))
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots-i_plots].set_xticklabels([])
            plt.tight_layout()   
     
        elif i_plots == 11:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+1-i_plots].set_xticks(month_starts)
            axs[i_plots+1-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+1-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+1-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+1-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                lns_late.extend(axs[i_plots+1-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4))
               
            axs[i_plots+1-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)                
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)      
            
            axs[i_plots+1-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+1-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+1-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+1-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
    
            axs[i_plots+1-i_plots].set_xticklabels([])
            axs[i_plots+1-i_plots].set_yticklabels([])
            plt.tight_layout()   
    
        elif i_plots == 12:
            formatter = FuncFormatter(millions)
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+2-i_plots].set_xticks(month_starts)
            axs[i_plots+2-i_plots].set_xticklabels(month_names)
            axs[i_plots+2-i_plots].set_ylabel(r'million km$^{3}$',fontsize=n_ylabel)
            
            axs[i_plots+2-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+2-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+2-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+2-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+2-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
         
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
                
            axs[i_plots+2-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+2-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+2-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+2-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+2-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots+2-i_plots].set_xticklabels([])
            axs[i_plots+2-i_plots].yaxis.set_major_formatter(formatter) 
            
            plt.tight_layout()  
    
        elif i_plots == 13:
            formatter = FuncFormatter(millions)
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+3-i_plots].set_xticks(month_starts)
            axs[i_plots+3-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+3-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+3-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+3-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+3-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+3-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
            
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots+3-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
    
            axs[i_plots+3-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+3-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots+3-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            
            axs[i_plots+3-i_plots].yaxis.set_major_formatter(formatter) 
            axs[i_plots+3-i_plots].set_xticklabels([])
            axs[i_plots+3-i_plots].set_yticklabels([])
            plt.tight_layout()  
    
    
        elif i_plots == 14:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+4-i_plots].set_xticks(month_starts)
            axs[i_plots+4-i_plots].set_xticklabels(month_names)
            axs[i_plots+4-i_plots].set_ylabel('DU',fontsize=n_ylabel)
            
            axs[i_plots+4-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+4-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+4-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+4-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+4-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
    
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
            
            axs[i_plots+4-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+4-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+4-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]]) 
            
            axs[i_plots+4-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+4-i_plots].tick_params(axis="y", labelsize=n_xyticks)
            axs[i_plots+4-i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout()    
    
    
        elif i_plots == 15:
            ix = [0,2]
            temp_per = np.zeros((2,365))
            for i in range(2):
                for j in range(365):
                    temp_per[i,j] = per_all_plot[i_plots][j][ix[i]]
                
            axs[i_plots+5-i_plots].set_xticks(month_starts)
            axs[i_plots+5-i_plots].set_xticklabels(month_names)
            
            axs[i_plots+5-i_plots].plot(mean_all_plot[i_plots], color = 'black', linewidth=4)
            axs[i_plots+5-i_plots].plot(max_all_plot[i_plots], color = 'grey', linewidth=4)
            axs[i_plots+5-i_plots].plot(min_all_plot[i_plots], color = 'grey', linewidth=4)
            
            for l6 in range(len_yea_early):
                axs[i_plots+5-i_plots].plot(new_data_plot[i_plots][np.where(len_years_red == sel_n_yea[i_plots][l6])[0][0]], color = sel_col[i_plots][l6], linewidth=4)
               
            axs[i_plots+5-i_plots].fill_between(len_days,temp_per[0],temp_per[1], facecolor = 'silver',edgecolor = 'lightblue', linewidth=4)
        
            props1 = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'grey',  linewidth=2)
    
            axs[i_plots+5-i_plots].text(sel_start_loc[i_plots],sel_end_loc[i_plots],r''+str(sel_txt[i_plots])+'',fontsize=n_txt,color = 'k',bbox= props1) 
            
            axs[i_plots+5-i_plots].set_ylim([sel_start_ylim[i_plots],sel_end_ylim[i_plots]])
            axs[i_plots+5-i_plots].set_xlim([sel_start_xlim[i_plots],sel_end_xlim[i_plots]])
            
            axs[i_plots+5-i_plots].yaxis.set_major_locator(MultipleLocator(sel_multi[i_plots]))
            axs[i_plots+5-i_plots].set_yticklabels([])
            axs[i_plots+5-i_plots].tick_params(axis="x", labelsize=n_xyticks)
            plt.tight_layout() 
    

    #make a new legend with all lines
    lns = lns_early+lns_late+lns_mean+lns_max+lns_min+lns_fill

    axs[0].legend(lns, leg_var_both, prop={'size': n_legend},loc='upper center', bbox_to_anchor=(1, 1.42),
                      ncol=3, fancybox=True, shadow=True)    
    
    
    plt.savefig(r''+str(path_out)+'final_quad6_'+str(model)+'_'+str(var_m_strat_T)+'_'+str(var_vpsc)+'_'+str(var_to3)+'_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300, bbox_inches='tight')
    plt.close()  
