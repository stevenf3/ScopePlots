import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import simps
from tkinter import Tk, filedialog
from mpl_axes_aligner import align
from math import ceil
import os

def pulse_detect(data, thresh):
    sign = data >= thresh
    pos1 = np.where(np.convolve(sign, [1, -1]) == 1)
    pos2 = np.where(np.convolve(sign, [1, -1]) == -1)
    return pos1[0][0], pos2[0][0]

root = Tk()
root.withdraw()
name1 = filedialog.askopenfilename(title='Choose First Run', filetypes=[('CSV Files', '.csv')])
name2 = filedialog.askopenfilename(title='Choose Second Run', filetypes=[('CSV Files', '.csv')])
basename1 = os.path.basename(os.path.normpath(name1))
basename2 = os.path.basename(os.path.normpath(name2))
print(basename1)
print(basename2)


frequency = 100
df1 = pd.read_csv(name1, skiprows=2, names=['time', 'Discharge Voltage', 'Discharge Current', 'Bias Voltage', 'Bias Current'])
df2 = pd.read_csv(name2, skiprows=2, names=['time', 'Discharge Voltage', 'Discharge Current', 'Bias Voltage', 'Bias Current'])
#print(df)


DI1 = df1['Discharge Current'].to_numpy()
DV1 = df1['Discharge Voltage'].to_numpy()
BI1 = df1['Bias Current'].to_numpy()
BV1 = df1['Bias Voltage'].to_numpy()

pulse_start, pulse_end = pulse_detect(DV1, 50)

DI2 = df2['Discharge Current'].to_numpy()
DV2 = df2['Discharge Voltage'].to_numpy()
BI2 = df2['Bias Current'].to_numpy()
BV2 = df2['Bias Voltage'].to_numpy()

t = df1['time'].to_numpy() * 1000000

timerange = t[pulse_end] - t[pulse_start]
timestep = timerange / (pulse_end-pulse_start)

DV1peakVal = simps(DV1[pulse_start:pulse_end], dx=timestep) / timerange
DV2peakVal = simps(DV2[pulse_start:pulse_end], dx=timestep) / timerange

DI1peakVal = simps(DI1[pulse_start:pulse_end], dx=timestep) / timerange
DI2peakVal = simps(DI2[pulse_start:pulse_end], dx=timestep) / timerange

BV1peakVal = simps(BV1[pulse_start:pulse_end], dx=timestep) / timerange
BV2peakVal = simps(BV2[pulse_start:pulse_end], dx=timestep) / timerange


BI1Val = simps(BI1, dx=timestep) / t[-1]
BI2Val = simps(BI2, dx=timestep) / t[-1]

DV1Val = simps(DV1, dx=timestep) / t[-1]
DV2Val = simps(DV2, dx=timestep) / t[-1]

DI1Val = simps(DI1, dx=timestep) / t[-1]
DI2Val = simps(DI2, dx=timestep) / t[-1]

BV1Val = simps(BV1, dx=timestep) / t[-1]
BV2Val = simps(BV2, dx=timestep) / t[-1]

BI1Val = simps(BI1, dx=timestep) / t[-1]
BI2Val = simps(BI2, dx=timestep) / t[-1]






print('voltage:', DV1peakVal)

maxDI1 = DI1.max()
minDI1 = DI1.min()
maxDV1 = DV1.max()
minDV1 = DV1.min()

maxBI1 = BI1.max()
minBI1 = BI1.min()
maxBV1 = BV1.max()
minBV1 = BV1.min()

maxDI2 = DI2.max()
minDI2 = DI2.min()
maxDV2 = DV2.max()
minDV2 = DV2.min()

maxBI2 = BI2.max()
minBI2 = BI2.min()
maxBV2 = BV2.max()
minBV2 = BV2.min()
#print(maxI, minI, maxV, minV)
filter_toggle = False

if filter_toggle == True:
    dcurrent1 = savgol_filter(DI1, 51, 3)
    dvoltage1 = savgol_filter(DV1, 51, 3)
    bcurrent1 = savgol_filter(BI1, 51, 3)
    bvoltage1 = savgol_filter(BV1, 51, 3)

    dcurrent2 = savgol_filter(DI2, 51, 3)
    dvoltage2 = savgol_filter(DV2, 51, 3)
    bcurrent2 = savgol_filter(BI2, 51, 3)
    bvoltage2 = savgol_filter(BV2, 51, 3)
else:
    dcurrent1 = DI1
    dvoltage1 = DV1
    bcurrent1 = BI1
    bvoltage1 = BV1

    dcurrent2 = DI2
    dvoltage2 = DV2
    bcurrent2 = BI2
    bvoltage2 = BV2

maxDI1 = dcurrent1.max()
minDI1 = dcurrent1.min()
maxDV1 = dvoltage1.max()
minDV1 = dvoltage1.min()
print(maxDV1)
maxBI1 = bcurrent1.max()
minBI1 = bcurrent1.min()
maxBV1 = bvoltage1.max()
minBV1 = bvoltage1.min()

maxDI2 = dcurrent2.max()
minDI2 = dcurrent2.min()
maxDV2 = dvoltage2.max()
minDV2 = dvoltage2.min()
print(maxDV2)
maxBI2 = bcurrent2.max()
minBI2 = bcurrent2.min()
maxBV2 = bvoltage2.max()
minBV2 = bvoltage2.min()


print(maxDV2, maxDI2)

plt.style.use('dark_background')
#plt.plot(t, dvoltage, color='yellow')
#plt.plot(t, dcurrent, color='limegreen')

'''
fig, ax = plt.subplots()

ax.set_title('Voltage and Current per pulse for ' + basename1[0:-4])
ax.set_ylabel('Voltage (V)')
ax.plot(t, dvoltage1, color='yellow')
ax.plot(t, bvoltage1, color='cyan')
ax.set_ylim(-(maxDV1)/2, maxDV1*1.2)

#ax.grid(color='gray', linestyle='dotted')

ax2 = ax.twinx()
ax2.set_ylabel('Current (A)')
ax2.plot(t, dcurrent1, color='limegreen')
ax2.plot(t, bcurrent1, color='fuchsia')

align.yaxes(ax, 0, ax2, 0, 0.2)

#plt.show()
'''

fig2, (axDV, axDI, axBV, axBI) = plt.subplots(4)
fig2.tight_layout()

#axDV = ax3[0,0]
#axDI = ax3[1,0]
#axBV = ax3[2,0]
#axBI = ax3[3,0]

axDV.plot(t, dvoltage1, color='yellow', label=basename1[0:-4])
axDV.plot(t, dvoltage2, color='orange', label=basename2[0:-4])
axDV.grid(color='gray', linestyle='dotted')
axDV.set_title('Discharge Voltage for ' + basename1[0:6] + ' and ' + basename2[0:6])
axDV.set_ylabel('Voltage (V)')
axDV.fill_between(t[pulse_start:pulse_end], min(minDV1, minDV2), max(maxDV1, maxDV2), alpha=0.3, facecolor='gray', interpolate=True)
axDV.text(t[ceil(pulse_start/4)], max(maxDV1, maxDV2)*(3/4), 'Voltage: '+ '{:.2f}'.format(DV1Val) +' V', color='yellow')
axDV.text(t[ceil(pulse_start/4)], max(maxDV1, maxDV2)*(2/4), 'Voltage: '+ '{:.2f}'.format(DV2Val) +' V', color='orange')
axDV.legend()

axDI.plot(t, dcurrent1, color='lime', label=basename1[0:-4])
axDI.plot(t, dcurrent2, color='seagreen', label=basename2[0:-4])
axDI.grid(color='gray', linestyle='dotted')
axDI.set_title('Discharge Current')
axDI.fill_between(t[pulse_start:pulse_end], min(minDI1, minDI2), max(maxDI1, maxDI2), alpha=0.3, facecolor='gray', interpolate=True)
axDI.text(t[ceil(pulse_start/4)], max(maxDI1, maxDI2)*(3/4), 'Current: '+ '{:.2f}'.format(DI1Val) +' A', color='lime')
axDI.text(t[ceil(pulse_start/4)], max(maxDI1, maxDI2)*(2/4), 'Current: '+ '{:.2f}'.format(DI2Val) +' A', color='seagreen')
axDI.set_ylabel('Current (A)')
axDI.legend()

axBV.plot(t, bvoltage1, color='cyan', label=basename1[0:-4])
axBV.plot(t, bvoltage2, color='darkcyan', label=basename2[0:-4])
axBV.grid(color='gray', linestyle='dotted')
axBV.set_title('Bias Voltage')
axBV.fill_between(t[pulse_start:pulse_end], min(minBV1, minBV2), max(maxBV1, maxBV2), alpha=0.3, facecolor='gray', interpolate=True)
axBV.text(t[ceil(pulse_start/4)], max(maxBV1, maxBV2)*(3/4), 'Voltage: '+ '{:.2f}'.format(BV1Val) +' V', color='cyan')
axBV.text(t[ceil(pulse_start/4)], max(maxBV1, maxBV2)*(2/4), 'Voltage: '+ '{:.2f}'.format(BV2Val) +' V', color='darkcyan')
axBV.set_ylabel('Voltage (V)')
axBV.legend()

axBI.plot(t, bcurrent1, color='fuchsia', label=basename1[0:-4])
axBI.plot(t, bcurrent2, color='darkviolet', label=basename2[0:-4])
axBI.grid(color='gray', linestyle='dotted')
axBI.set_title('Bias Current')
axBI.fill_between(t[pulse_start:pulse_end], min(minBI1, minBI2), max(maxBI1, maxBI2), alpha=0.3, facecolor='gray', interpolate=True)
axBI.text(t[ceil(pulse_start/4)], max(minBI1, minBI2)*(2/4), 'Current: '+ '{:.2f}'.format(BI1Val) +' A', color='fuchsia')
axBI.text(t[ceil(pulse_start/4)], max(minBI1, minBI2)*(3/5), 'Current: '+ '{:.2f}'.format(BI2Val) +' A', color='darkviolet')
axBI.set_ylabel('Current (A)')
axBI.set_xlabel('Time (µs)')
#axBI.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
axBI.legend()

plt.show()