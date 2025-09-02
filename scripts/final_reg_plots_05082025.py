# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 22:31:16 2025

@author: Jevare
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob
import os
import pandas as pd 
from datetime import datetime

from matplotlib.ticker import MultipleLocator



def Linear_Regm(periods,ozone_march):
    r_sq = []
    var_per = np.array(periods).reshape(len(len_years_red))
    var_ozone_march = np.array(ozone_march).reshape(len(len_years_red))  
    x = var_per[:].reshape((-1,1))
    y = var_ozone_march[:]
    model = LinearRegression()
    model.fit(x, y)
    r_sq.append(model.score(x, y))
    y_pred = model.predict(x)
    #print(f"predicted response:\n{y_pred}")    
    return(r_sq,x,y,y_pred)
    
    

len_years = np.linspace(1980,2024,45).astype(int)
len_years_red = np.linspace(1980,2023,44).astype(int)

model = 'ERA5' #ERA5 or MERRA2 

path_in = 'C:/...' #location, where your data is stored
path_out = 'C:/...' #location of your output folder

path = ''+str(path_in)+''

all_files = glob.glob(os.path.join(path, 'final_'+str(model)+'_all*.csv'))

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df = pd.concat(df_from_each_file, ignore_index=False, axis = 1)

var_txt = ['Temperature_50hPa','Temperature_30hPa','O3_50hPa','O3_30hPa','TO3','Apsc_50hPa','Apsc_30hPa','Vpsc','vT','m_strat_T', 'Day', 'Month', 'Year']
n = np.size(var_txt)+1 # drop the index column and choose just every variable columns e.g. above
val_new = []
val_rev = []

var_to3 = 'TO3'
var_Vpsc = 'Vpsc'
var_vT = 'vT'
var_m_strat_T = 'm_strat_T'

var_title = '30hPa'

mean_march_to3 = []

mean_to3_first_period = []
mean_to3_second_period = []
mean_to3_third_period = []
mean_to3_seventh_period = []

mean_Vpsc_first_period = []
mean_Vpsc_second_period = []
mean_Vpsc_third_period = []
mean_Vpsc_seventh_period = []


mean_vT_first_period = []
mean_vT_second_period = []
mean_vT_third_period = []
mean_vT_seventh_period = []

mean_m_strat_T_first_period = []
mean_m_strat_T_second_period = []
mean_m_strat_T_third_period = []
mean_m_strat_T_seventh_period = []


mean_all_to3_march = []

mean_all_to3_first_period = []
mean_all_to3_second_period = []
mean_all_to3_third_period = []
mean_all_to3_seventh_period = []


mean_all_Vpsc_first_period = []
mean_all_Vpsc_second_period = []
mean_all_Vpsc_third_period = []
mean_all_Vpsc_seventh_period = []

mean_all_vT_first_period = []
mean_all_vT_second_period = []
mean_all_vT_third_period = []
mean_all_vT_seventh_period = []

mean_all_m_strat_T_first_period = []
mean_all_m_strat_T_second_period = []
mean_all_m_strat_T_third_period = []
mean_all_m_strat_T_seventh_period = []


len_var_sel = 31
len_to3 = 31


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

#---------------------    
st_1stmar = 243
st_15tdec_14thjan = 167
st_15thjan_14thfeb = 198
st_28thjan_27thfeb = 211
st_14thfeb_16thmar = 228

for i in range(len(val_rev)):
    mean_march_to3.append(np.nanmean(val_rev[i][var_to3][st_1stmar:st_1stmar+len_to3].astype(float))) # March mean

    mean_to3_first_period.append(np.nanmean(val_rev[i][var_to3][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float))) # 15th of Dec till 14th of Jan
    mean_to3_second_period.append(np.nanmean(val_rev[i][var_to3][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float))) # 15th of Jan till 14th of Feb
    mean_to3_third_period.append(np.nanmean(val_rev[i][var_to3][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float))) # 28th of Jan till 27th of Feb        
    mean_to3_seventh_period.append(np.nanmean(val_rev[i][var_to3][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float))) # 14th of Feb till 16th of Mar 
  
    mean_Vpsc_first_period.append(np.nanmean(val_rev[i][var_Vpsc][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float))) # 15th of Dec till 14th of Jan
    mean_Vpsc_second_period.append(np.nanmean(val_rev[i][var_Vpsc][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float))) # 15th of Jan till 14th of Feb
    mean_Vpsc_third_period.append(np.nanmean(val_rev[i][var_Vpsc][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float))) # 28th of Jan till 27th of Feb   
    mean_Vpsc_seventh_period.append(np.nanmean(val_rev[i][var_Vpsc][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float))) # 14th of Feb till 16th of Mar   
     
    
    mean_vT_first_period.append(np.nanmean(val_rev[i][var_vT][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float))) # 15th of Dec till 14th of Jan
    mean_vT_second_period.append(np.nanmean(val_rev[i][var_vT][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float))) # 15th of Jan till 14th of Feb
    mean_vT_third_period.append(np.nanmean(val_rev[i][var_vT][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float))) # 28th of Jan till 27th of Feb   
    mean_vT_seventh_period.append(np.nanmean(val_rev[i][var_vT][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float))) # 14th of Feb till 16th of Mar    
    
    
    mean_m_strat_T_first_period.append(np.nanmean(val_rev[i][var_m_strat_T][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float))) # 15th of Dec till 14th of Jan
    mean_m_strat_T_second_period.append(np.nanmean(val_rev[i][var_m_strat_T][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float))) # 15th of Jan till 14th of Feb
    mean_m_strat_T_third_period.append(np.nanmean(val_rev[i][var_m_strat_T][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float))) # 28th of Jan till 27th of Feb   
    mean_m_strat_T_seventh_period.append(np.nanmean(val_rev[i][var_m_strat_T][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float))) # 14th of Feb till 16th of Mar    
    
    
   
    
    for n_days_fip in range(st_15tdec_14thjan,st_15tdec_14thjan+len_var_sel):
        mean_all_to3_first_period.append(np.float64(val_rev[i][var_to3][n_days_fip]))
        mean_all_Vpsc_first_period.append(np.float64(val_rev[i][var_Vpsc][n_days_fip]))
        mean_all_vT_first_period.append(np.float64(val_rev[i][var_vT][n_days_fip]))
        mean_all_m_strat_T_first_period.append(np.float64(val_rev[i][var_m_strat_T][n_days_fip]))
        
    for n_days_sep in range(st_15thjan_14thfeb,st_15thjan_14thfeb+len_var_sel):
        mean_all_to3_second_period.append(np.float64(val_rev[i][var_to3][n_days_sep]))
        mean_all_Vpsc_second_period.append(np.float64(val_rev[i][var_Vpsc][n_days_sep]))
        mean_all_vT_second_period.append(np.float64(val_rev[i][var_vT][n_days_sep]))
        mean_all_m_strat_T_second_period.append(np.float64(val_rev[i][var_m_strat_T][n_days_sep]))
        
    for n_days_thp in range(st_28thjan_27thfeb,st_28thjan_27thfeb+len_var_sel):
        mean_all_to3_third_period.append(np.float64(val_rev[i][var_to3][n_days_thp]))
        mean_all_Vpsc_third_period.append(np.float64(val_rev[i][var_Vpsc][n_days_thp]))
        mean_all_vT_third_period.append(np.float64(val_rev[i][var_vT][n_days_thp]))
        mean_all_m_strat_T_third_period.append(np.float64(val_rev[i][var_m_strat_T][n_days_thp]))

    for n_days_sev in range(st_14thfeb_16thmar,st_14thfeb_16thmar+len_var_sel):
        mean_all_to3_seventh_period.append(np.float64(val_rev[i][var_to3][n_days_sev]))
        mean_all_Vpsc_seventh_period.append(np.float64(val_rev[i][var_Vpsc][n_days_sev]))
        mean_all_vT_seventh_period.append(np.float64(val_rev[i][var_vT][n_days_sev]))
        mean_all_m_strat_T_seventh_period.append(np.float64(val_rev[i][var_m_strat_T][n_days_sev]))
        
        
    for n_days_march in range(st_1stmar,st_1stmar+len_to3):
        mean_all_to3_march.append(np.float64(val_rev[i][var_to3][n_days_march]))
        
#----------------------------------------------------------------  
        
predm_to3_fip = Linear_Regm(mean_to3_first_period,mean_march_to3)
predm_to3_sep = Linear_Regm(mean_to3_second_period,mean_march_to3)
predm_to3_thp = Linear_Regm(mean_to3_third_period,mean_march_to3)
predm_to3_sev = Linear_Regm(mean_to3_seventh_period,mean_march_to3)

predm_Vpsc_fip = Linear_Regm(mean_Vpsc_first_period,mean_march_to3)
predm_Vpsc_sep = Linear_Regm(mean_Vpsc_second_period,mean_march_to3)
predm_Vpsc_thp = Linear_Regm(mean_Vpsc_third_period,mean_march_to3)
predm_Vpsc_sev = Linear_Regm(mean_Vpsc_seventh_period,mean_march_to3)

predm_vT_fip = Linear_Regm(mean_vT_first_period,mean_march_to3)
predm_vT_sep = Linear_Regm(mean_vT_second_period,mean_march_to3)
predm_vT_thp = Linear_Regm(mean_vT_third_period,mean_march_to3)
predm_vT_sev = Linear_Regm(mean_vT_seventh_period,mean_march_to3)

predm_m_strat_T_fip = Linear_Regm(mean_m_strat_T_first_period,mean_march_to3)
predm_m_strat_T_sep = Linear_Regm(mean_m_strat_T_second_period,mean_march_to3)
predm_m_strat_T_thp = Linear_Regm(mean_m_strat_T_third_period,mean_march_to3)
predm_m_strat_T_sev = Linear_Regm(mean_m_strat_T_seventh_period,mean_march_to3)


#print(predm_vT_sev[0])

#Plots

a = 335

title_txt = [val_rev[0]['Day'][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2)+'/'+val_rev[0]['Month'][st_15tdec_14thjan:st_15tdec_14thjan+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2),
             val_rev[0]['Day'][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2)+'/'+val_rev[0]['Month'][st_15thjan_14thfeb:st_15thjan_14thfeb+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2),
             val_rev[0]['Day'][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2)+'/'+val_rev[0]['Month'][st_28thjan_27thfeb:st_28thjan_27thfeb+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2),
             val_rev[0]['Day'][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2)+'/'+val_rev[0]['Month'][st_14thfeb_16thmar:st_14thfeb_16thmar+len_var_sel].astype(float).astype(int).astype(str).str.zfill(2)]                          
         

fig, axs = plt.subplots(4,1,figsize=(7,12))
axs[0].plot(len_years_red,mean_march_to3, marker = '.', linestyle ='-', color='black')
axs[0].plot(len_years_red,predm_m_strat_T_fip[3], marker = '.', linestyle ='-',color='red')
axs[0].plot(len_years_red,predm_to3_fip[3], marker = '.', linestyle ='-',color='green')
axs[0].plot(len_years_red,predm_Vpsc_fip[3], marker = '.', linestyle ='-',color='blue')
axs[0].plot(len_years_red,predm_vT_fip[3], marker = '.', linestyle ='-',color='darkviolet', markersize = 4)

axs[0].set_title(''+str(title_txt[0][st_15tdec_14thjan])+' - '+title_txt[0][st_15tdec_14thjan+len_var_sel-1]+'')

axs[0].text(len_years_red[0],455,r'$\bf{'+str(model)+'}$' '\n' '(a)', color = 'black')

axs[0].text(len_years_red[0],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_m_strat_T_fip[0]),2))+'', color = 'red')
axs[0].text(len_years_red[10],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_to3_fip[0]),2))+'', color = 'green')
axs[0].text(len_years_red[20],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_Vpsc_fip[0]),2))+'', color = 'blue')
axs[0].text(len_years_red[30],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_vT_fip[0]),2))+'', color = 'darkviolet')

axs[0].set_ylim([330,480])
axs[0].xaxis.set_major_locator(MultipleLocator(10))
axs[0].set_xlim([1979.5,2023.5])
axs[0].yaxis.set_major_locator(MultipleLocator(30))


axs[1].plot(len_years_red,mean_march_to3, marker = '.', linestyle ='-', color='black')
axs[1].plot(len_years_red,predm_m_strat_T_sep[3], marker = '.', linestyle ='-',color='red')
axs[1].plot(len_years_red,predm_to3_sep[3], marker = '.', linestyle ='-',color='green')
axs[1].plot(len_years_red,predm_Vpsc_sep[3], marker = '.', linestyle ='-',color='blue')
axs[1].plot(len_years_red,predm_vT_sep[3], marker = '.', linestyle ='-',color='darkviolet', markersize = 4)

axs[1].set_title(''+str(title_txt[1][st_15thjan_14thfeb])+' - '+title_txt[1][st_15thjan_14thfeb+len_var_sel-1]+'')

axs[1].text(len_years_red[0],467,r'(b)', color = 'black')

axs[1].text(len_years_red[0],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_m_strat_T_sep[0]),2))+'', color = 'red')
axs[1].text(len_years_red[10],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_to3_sep[0]),2))+'', color = 'green')
axs[1].text(len_years_red[20],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_Vpsc_sep[0]),2))+'', color = 'blue')
axs[1].text(len_years_red[30],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_vT_sep[0]),2))+'', color = 'darkviolet')

axs[1].set_ylim([330,480])
axs[1].xaxis.set_major_locator(MultipleLocator(10))
axs[1].set_xlim([1979.5,2023.5])
axs[1].yaxis.set_major_locator(MultipleLocator(30))


axs[2].plot(len_years_red,mean_march_to3, marker = '.', linestyle ='-', color='black')
axs[2].plot(len_years_red,predm_m_strat_T_thp[3], marker = '.', linestyle ='-',color='red')
axs[2].plot(len_years_red,predm_to3_thp[3], marker = '.', linestyle ='-',color='green')
axs[2].plot(len_years_red,predm_Vpsc_thp[3], marker = '.', linestyle ='-',color='blue')
axs[2].plot(len_years_red,predm_vT_thp[3], marker = '.', linestyle ='-',color='darkviolet', markersize = 4)

axs[2].set_title(''+str(title_txt[2][st_28thjan_27thfeb])+' - '+title_txt[2][st_28thjan_27thfeb+len_var_sel-1]+'')

axs[2].text(len_years_red[0],467,r'(c)', color = 'black')

axs[2].text(len_years_red[0],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_m_strat_T_thp[0]),2))+'', color = 'red')
axs[2].text(len_years_red[10],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_to3_thp[0]),2))+'', color = 'green')
axs[2].text(len_years_red[20],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_Vpsc_thp[0]),2))+'', color = 'blue')
axs[2].text(len_years_red[30],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_vT_thp[0]),2))+'', color = 'darkviolet')

axs[2].set_ylim([330,480])
axs[2].xaxis.set_major_locator(MultipleLocator(10))
axs[2].set_xlim([1979.5,2023.5])
axs[2].yaxis.set_major_locator(MultipleLocator(30))



axs[3].plot(len_years_red,mean_march_to3, marker = '.', linestyle ='-', color='black')
axs[3].plot(len_years_red,predm_m_strat_T_sev[3], marker = '.', linestyle ='-',color='red')
axs[3].plot(len_years_red,predm_to3_sev[3], marker = '.', linestyle ='-',color='green')
axs[3].plot(len_years_red,predm_Vpsc_sev[3], marker = '.', linestyle ='-',color='blue')
axs[3].plot(len_years_red,predm_vT_sev[3], marker = '.', linestyle ='-',color='darkviolet', markersize = 4)

axs[3].set_title(''+str(title_txt[3][st_14thfeb_16thmar])+' - '+title_txt[3][st_14thfeb_16thmar+len_var_sel-1]+'')

axs[3].text(len_years_red[0],467,r'(d)', color = 'black')

axs[3].text(len_years_red[0],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_m_strat_T_sev[0]),2))+'', color = 'red')
axs[3].text(len_years_red[10],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_to3_sev[0]),2))+'', color = 'green')
axs[3].text(len_years_red[20],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_Vpsc_sev[0]),2))+'', color = 'blue')
axs[3].text(len_years_red[30],a, r'R$^{2}$ = '+str(np.round(np.nanmean(predm_vT_sev[0]),2))+'', color = 'darkviolet')

axs[3].set_ylim([330,480])
axs[3].xaxis.set_major_locator(MultipleLocator(10))
axs[3].set_xlim([1979.5,2023.5])
axs[3].yaxis.set_major_locator(MultipleLocator(30))


for ax in axs.flat:
    ax.set(ylabel='TCO [DU]')
    
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


ax.legend(['OBS $\mathregular{TCO_{M}}$', '$\mathregular{TCO_{M}}$ ~ $\mathregular{T_{STRAT}}$','$\mathregular{TCO_{M}}$ ~ TCO','$\mathregular{TCO_{M}}$ ~ $\mathregular{V_{PSC}}$','$\mathregular{TCO_{M}}$ ~ EHF'], loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=2)

leg = ax.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('red')
leg.legendHandles[2].set_color('green')
leg.legendHandles[3].set_color('blue')
leg.legendHandles[4].set_color('darkviolet')

plt.savefig(r''+str(path_out)+'final_vert_all_'+str(model)+'_'+datetime.today().strftime('%d%m%Y')+'.png',dpi=300,bbox_inches='tight') 
plt.close() 