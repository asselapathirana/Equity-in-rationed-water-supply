"""
=====================================================================================================
Gini index and Available Demand Fraction Calculations 
=====================================================================================================
   
   .. Copyright 2019,2020 Assela Pathirana

    .. This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

    .. This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    .. You should have received a copy of the GNU General Public License
       along with this program.  If not, see <http://www.gnu.org/licenses/>.
       
    .. author:: Assela Pathirana <assela@pathirana.net>

"""
import re
import numpy as np
import wntr
import pandas as pd
import ineqpy
import matplotlib.pyplot as plt
from functools import reduce
from math import gcd

#pip install git+https://github.com/mmngreco/IneqPy.git

#input data BEGIN ####################################################################
# note: To avoid inaccuracies, make sure reporting_duration is a whole multiple of each cyclelen
cyclelen=[12, 24, 48, 72, 144, 288, 576] # 
reporting_duration=576
rationingtimefactor = 0.75

source_junction='J-1'

# Input (epanet network format) file name
inp_filename = './Base_model_withFCV.inp' 
out_locname = './networks/{}'.format(inp_filename) 
#input data END ######################################################################

# IMPORTANT: We assume reporting time step of 1h!!! 
FORMAT = "{:>50} : {:10.4f}"
pattern = re.compile("^.*__demand__[0-9]$")
SEC2H = 3600

outname='{}_{:.2f}perc_ratining'.format(out_locname[:-4],rationingtimefactor)


lcm = reduce(lambda a,b: a*b // gcd(a,b),cyclelen)
if reporting_duration % lcm:
    print("Problem: {} has lcm of {} and {} is not a multiple of that!".format(cyclelen, lcm, reporting_duration))
    exit(1)    
else:
    print ("Using simulation length as {}h (lcm={})".format(reporting_duration, lcm))
for cy in cyclelen:
    if int(rationingtimefactor*cy)!=rationingtimefactor*cy:
        print("One of the cyclelengths result in non-integer step {}x{}={}".format(cy,rationingtimefactor,cy*rationingtimefactor))
        exit(1)

extratime=max(cyclelen) # this period, we'll ignore in calculations of ADF, gini index etc. 
simulation_duration = reporting_duration + extratime #


def gini_index(wn, dll, results=None):
    results = run_sim(wn,results)
    # Calculate 
    expected_demand = wntr.metrics.expected_demand(wn)[dll]
    demand = results.link['flowrate'][dll]
    expected_demand=subset(expected_demand,demand)
    expected_demand=expected_demand.sum()
    demand = demand.sum()
    # create a dataframe we use expected demand as weight
    data = pd.concat([expected_demand.rename('ed'),demand.rename('adf')],axis=1, sort=False)
    data['adf']=data['adf']/data['ed']
    gi=ineqpy.gini('adf', weights='ed',data=data)
    lo=ineqpy.lorenz('adf', weights='ed', data=data)
    return gi, lo

def subset(df1,df2):
    partn = df2.shape[0] 
    # take the last partn columns from df1
    return df1.tail(partn) #df1[:-partn]


def water_service_avail(wn, dll, results=None):
    results = run_sim(wn,results)
    # Calculate ADF 
    expected_demand = wntr.metrics.expected_demand(wn)[dll]
    demand = results.link['flowrate'][dll]   
    expected_demand=subset(expected_demand,demand)
    
    
    tl=results.node['pressure'][['T1','T2']]
    all=pd.concat([demand, expected_demand, tl], axis=1, join='inner')
    wsa1=demand.sum().sum()/expected_demand.sum().sum()
    return wsa1 

def run_sim(wn, results=None):
    if not results:
        # Run a Pressure Dependant Demand simulation 
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim() 
    return results
    
wn = wntr.network.WaterNetworkModel(out_locname) 

# filter the links supplying demand nodes
dll=[x[0] for x in list(wn.links()) if pattern.match(x[0])]

plt.figure(figsize=[6.4*2, 4.8])
ax=plt.subplot(1,2,1)


zero_and_cyclelen=[0]+cyclelen

names=["Cycle_{}h".format(x) if x>0 else "Continuous" for x in zero_and_cyclelen]
adfs=[]
ginis=[]
lors=[]
for i, cycle in enumerate(zero_and_cyclelen):
    # Reset all just in case
    wn.reset_initial_values()
    # set simulation duration
    wn.options.time.duration = simulation_duration*SEC2H
    wn.options.time.report_timestep=SEC2H
    wn.options.time.report_start=SEC2H*extratime
    # set pattern
    step_size=SEC2H
    start_time=rationingtimefactor*SEC2H*cycle 
    end_time=SEC2H*cycle
    duration=SEC2H*cycle
    if(cycle==0):
        end_time=duration=SEC2H*24 # duration does not matter - we don't cut water
    pat=wntr.network.elements.Pattern.binary_pattern(names[i], 
                                                     step_size=step_size, # hourly step
                                                     start_time=start_time, 
                                                     end_time=end_time,
                                                     duration=duration,
                                                     wrap=True,
                                                     )
    
    print("pattern: step_size={}h, start_time={}h, end_time={}h, duration={}h".format(step_size/SEC2H,start_time/SEC2H,end_time/SEC2H, duration/SEC2H))
    nn=wn.get_node(source_junction) 
    bd=nn.demand_timeseries_list[0].base_value#get the base demand
    nn.demand_timeseries_list.clear() # remove any patterns
    wn.add_pattern(pat.name,pat)
    nn.demand_timeseries_list.append((bd,pat,pat.name))
    
    outputfilename='{}_{:.2f}prec_rationing_{}h_cycle.inp'.format(out_locname[:-4],int(100*rationingtimefactor),cycle)
    wn.write_inpfile(outputfilename)
    
    results=run_sim(wn)
    wsa1=water_service_avail(wn, dll, results=results)
    gini, lor = gini_index(wn, dll, results=results)
    lor=pd.DataFrame({pat.name: lor.values[:, 1]}, index=list(lor.index))
    print(FORMAT.format("Water service availability:",wsa1))
    print(FORMAT.format("gini_index:", gini))
    adfs.append(wsa1)
    ginis.append(gini)

    ax=lor.plot(ax=ax)
    ax.set_xlim(xmin=0.0, xmax=1.0)
    ax.set_xlabel("Cumulative cutomer fraction (-)")
    ax.set_ylabel("Cumulative ADF distribution (-)")

    nam=lor.columns[0]
    lor[nam+"_x"]=lor.index
    lor=lor.rename({nam:nam+"_y"})
    col=lor.columns.tolist()
    lor=lor[reversed(col)].reset_index(drop=True)
    
    lors.append(lor)

lorenzs=pd.concat(lors,axis=1)
lorenzs.to_excel(out_locname[:-4]+"_lorenz.xlsx")

ax=plt.subplot(1,2,2)
plt.plot(adfs[1:],ginis[1:])
ax.set_xlabel("ADF (-)")
ax.set_ylabel("Gini index (-)")
for i, txt in enumerate(names[1:]):
    ax.annotate(txt, (adfs[1:][i], ginis[1:][i]))
plt.savefig(out_locname[:-4]+".png")
plt.show()
print("End")
