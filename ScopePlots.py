import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import simps
from tkinter import Tk, filedialog
from mpl_axes_aligner import align
import os


def rising_edge(data, thresh):
    sign = data >= thresh
    pos1 = np.where(np.convolve(sign, [1, -1]) == 1)
    pos2 = np.where(np.convolve(sign, [1, -1]) == -1)
    return [pos1[0][0], pos2[0][0]]


root = Tk()
root.withdraw()
name = filedialog.askopenfilename(title='Choose a File', filetypes=[('CSV Files', '.csv')])
basename = os.path.basename(os.path.normpath(name))
print(basename)


frequency = 100
df = pd.read_csv(name, skiprows=2, names=['time', 'Discharge Voltage', 'Discharge Current', 'Bias Voltage', 'Bias Current'])
print(df)


DI = df['Discharge Current'].to_numpy()
DV = df['Discharge Voltage'].to_numpy()
BI = df['Bias Current'].to_numpy()
BV = df['Bias Voltage'].to_numpy()
t = df['time'].to_numpy()

print(rising_edge(DV, 50)[0])

maxDI = DI.max()
minDI = DI.min()
maxDV = DV.max()
minDV = DV.min()

maxBI = BI.max()
minBI = BI.min()
maxBV = BV.max()
minBV = BV.min()
#print(maxI, minI, maxV, minV)
filter_toggle = True

if filter_toggle == True:
    dcurrent = savgol_filter(DI, 51, 3)
    dvoltage = savgol_filter(DV, 51, 3)
    bcurrent = savgol_filter(BI, 51, 3)
    bvoltage = savgol_filter(BV, 51, 3)
else:
    dcurrent = DI
    dvoltage = DV
    bcurrent = BI
    bvoltage = BV

maxDI = dcurrent.max()
minDI = dcurrent.min()
maxDV = dvoltage.max()
minDV = dvoltage.min()
print(maxDV)
maxBI = bcurrent.max()
minBI = bcurrent.min()
maxBV = bvoltage.max()
minBV = bvoltage.min()

print(maxDV, maxDI)

plt.style.use('dark_background')
#plt.plot(t, dvoltage, color='yellow')
#plt.plot(t, dcurrent, color='limegreen')
fig, ax = plt.subplots()

ax.set_title('Voltage and Current per pulse for ' + basename[0:-4])
ax.set_ylabel('Voltage (V)')
ax.plot(t, dvoltage, color='yellow')
ax.plot(t, bvoltage, color='cyan')
ax.set_ylim(-(maxDV)/2, maxDV*1.2)

#ax.grid(color='gray', linestyle='dotted')

ax2 = ax.twinx()
ax2.set_ylabel('Current (A)')
ax2.plot(t, dcurrent, color='limegreen')
ax2.plot(t, bcurrent, color='fuchsia')

align.yaxes(ax, 0, ax2, 0, 0.2)

#plt.show()

fig2, (axDV, axDI, axBV, axBI) = plt.subplots(4)
fig2.tight_layout()

#axDV = ax3[0,0]
#axDI = ax3[1,0]
#axBV = ax3[2,0]
#axBI = ax3[3,0]

axDV.plot(t, dvoltage, color='yellow')
axDV.grid(color='gray', linestyle='dotted')
axDV.set_title('Discharge Voltage for ' + basename[0:-4])
axDV.set_ylabel('Voltage (V)')

axDI.plot(t, dcurrent, color='limegreen')
axDI.grid(color='gray', linestyle='dotted')
axDI.set_title('Discharge Current')
axDI.set_ylabel('Current (A)')

axBV.plot(t, bvoltage, color='cyan')
axBV.grid(color='gray', linestyle='dotted')
axBV.set_title('Bias Voltage')
axBV.set_ylabel('Voltage (V)')

axBI.plot(t, bcurrent, color='fuchsia')
axBI.grid(color='gray', linestyle='dotted')
axBI.set_title('Bias Current')
axBI.set_ylabel('Current (A)')
axBI.set_xlabel('Time (s)')

plt.show()