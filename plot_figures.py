import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from fnmatch import fnmatch
import seaborn as sns
import pickle
import os



#plot result of Figure 3

list_widths=[4,8,16,32,64,128,256,512,1024,2048]
list_depths=[3,4,5,6,7,8,9,10,11,12,13]



the_folder1 = "./CodeR_BTmodel/result"
the_folder2 = "./CodeR_THmodel/result"
list_allfile1 = os.listdir(the_folder1)
list_allfile2 = os.listdir(the_folder2)
Regret1 = np.zeros([len(list_depths),len(list_widths)])
Regret2 = np.zeros([len(list_depths),len(list_widths)])

for i in range(len(list_depths)):
    for j in range(len(list_widths)):
        The_DATA_MARK = '*dCOR_idata_The_widths_%s_depths_%s_*' %(list_widths[j],list_depths[i])
        list_file = []
        for ifile in list_allfile1:
            if fnmatch(ifile,The_DATA_MARK):
                list_file.append(ifile)
        len(list_file)
        the_res = []
        for idx_data in range(len(list_file)):
            with open(os.path.join(the_folder1,  list_file[idx_data]), 'rb') as file:
                ff = pickle.load(file)
                the_res.append(ff['reg_test'].item())
        Regret1[i,j] = np.mean(the_res)

for i in range(len(list_depths)):
    for j in range(len(list_widths)):
        The_DATA_MARK = '*dCOR_idata_The_widths_%s_depths_%s_*' %(list_widths[j],list_depths[i])
        list_file = []
        for ifile in list_allfile2:
            if fnmatch(ifile,The_DATA_MARK):
                list_file.append(ifile)
        len(list_file)
        the_res = []
        for idx_data in range(len(list_file)):
            with open(os.path.join(the_folder2,  list_file[idx_data]), 'rb') as file:
                ff = pickle.load(file)
                the_res.append(ff['reg_test'].item())
        Regret2[i,j] = np.mean(the_res)
# Original data points

Result=list([np.array(Regret1),np.array(Regret2)])




width_log = np.log2(list_widths)
X, Y = np.meshgrid(list_depths, width_log)

# Create finer grid for smooth visualization
depths_fine = np.linspace(min(list_depths), max(list_depths), 50)
width_log_fine = np.linspace(min(width_log), max(width_log), 50)
X_fine, Y_fine = np.meshgrid(depths_fine, width_log_fine)

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
axes = []

vmin = min(Result[0].min(), Result[1].min())
vmax = max(Result[0].max(), Result[1].max())

for j in range(2):
    ax = fig.add_subplot(gs[0, j], projection='3d')
    axes.append(ax)
axes = np.array(axes)

titles = [r"BT", r"Thurstonian"]

for idx, values in enumerate(Result):
    ax = axes[idx]
    Z = values.T
    
    # Create spline interpolation
    # Adjust smoothing factor s to control smoothness
    spline = RectBivariateSpline(width_log, list_depths, Z, 
                                kx=3, ky=3,  # cubic spline
                                s=0.001)
    
    # Evaluate spline on fine grid
    Z_fine = spline(width_log_fine, depths_fine)
    
    surf = ax.plot_surface(X_fine, Y_fine, Z_fine, 
                          cmap='coolwarm',
                          vmin=vmin, vmax=vmax,
                          antialiased=True)
    vmin = min(Result[0].min(), Result[1].min())
    vmax = max(Result[0].max(), Result[1].max())
    ax.view_init(elev=20, azim=15)
    ax.set_xlabel('depth', fontsize=10)
    ax.set_ylabel(r'$\operatorname{log}_2(\text{width})$', fontsize=14)
    ax.set_zlabel('regret', labelpad=5, fontsize=14)
    ax.set_title(titles[idx], pad=-10, fontsize=14)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))


cax = fig.add_subplot(gs[0, -1])
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, shrink=0.8, aspect=10)
cax.set_position([cax.get_position().x0 - 0.02,
                 cax.get_position().y0 + 0.1,
                 cax.get_position().width * 0.6,
                 cax.get_position().height * 0.8])

plt.savefig(f"Two_Regrets.png", dpi=600)
plt.show()


#plot result of Figure 4
noise_level=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
depth= 4
width = 64
sim=50
list_sim=[i for i in range(sim)]

Regret3 = np.zeros([len(noise_level),sim])
for i in range(len(noise_level)):
    for r in range(len(list_sim)):

        file_path = './CodeR_BTmodel/result/dCOR_idata_The_widths_%s_depths_%s_band_%s_sim_%s__res.pickle' %(width,depth,noise_level[i],list_sim[r])
        try:
            with open(file_path, 'rb') as file:
                Regret3[i,r] = pickle.load(file)['reg_test'].item()
                    
        except (FileNotFoundError, KeyError):
            Regret3[i,  r] = np.nan
            
data = []
for i in range(Regret3.shape[0]):
    for j in range(Regret3.shape[1]):
        data.append({'regret': Regret3[i, j], 'Band': noise_level[i]})
df = pd.DataFrame(data)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pal = sns.cubehelix_palette(len(noise_level), rot=-.25, light=.7)
g = sns.FacetGrid(df, row="Band", hue="Band",row_order=sorted(noise_level, reverse=True), aspect=10, height=.5, palette=pal)

g.map(sns.kdeplot, "regret", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=0.5)
g.map(sns.kdeplot, "regret", clip_on=False, color="w", lw=2, bw_adjust=.5)

g.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)

g.figure.subplots_adjust(hspace=-.25)
g.set_titles("")

for k in range(len(noise_level)):
    g.axes[k, 0].set(yticks=[])
    ylabel = g.axes[k, 0].set_ylabel(f"{noise_level[k]:.2f}" , fontsize=12, rotation=0,loc='bottom')  # Set ylabel

text_str = "noise level"
plt.text(0.08, 0.5, text_str, fontsize=12, ha='center', va='center', 
         transform=g.fig.transFigure, rotation=90)

g.despine(bottom=True, left=True)

plt.tight_layout()
plt.savefig("bt_margin.png", dpi=600)



Regret4 = np.zeros([len(noise_level),sim])
for i in range(len(noise_level)):
    for r in range(len(list_sim)):

        file_path = './CodeR_THmodel/result/dCOR_idata_The_widths_%s_depths_%s_band_%s_sim_%s__res.pickle' %(width,depth,noise_level[i],list_sim[r])
        try:
            with open(file_path, 'rb') as file:
                Regret4[i,r] = pickle.load(file)['reg_test'].item()
                    
        except (FileNotFoundError, KeyError):
            Regret4[i,  r] = np.nan
            
data = []
for i in range(Regret4.shape[0]):
    for j in range(Regret4.shape[1]):
        data.append({'regret': Regret4[i, j], 'Band': noise_level[i]})
df = pd.DataFrame(data)

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

pal = sns.cubehelix_palette(len(noise_level), rot=-.25, light=.7)
g = sns.FacetGrid(df, row="Band", hue="Band",row_order=sorted(noise_level, reverse=True), aspect=10, height=.5, palette=pal)

g.map(sns.kdeplot, "regret", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=0.5)
g.map(sns.kdeplot, "regret", clip_on=False, color="w", lw=2, bw_adjust=.5)

g.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)

g.figure.subplots_adjust(hspace=-.25)
g.set_titles("")

for k in range(len(noise_level)):
    g.axes[k, 0].set(yticks=[])
    ylabel = g.axes[k, 0].set_ylabel(f"{noise_level[k]:.2f}" , fontsize=12, rotation=0,loc='bottom')  # Set ylabel

text_str = "noise level"
plt.text(0.08, 0.5, text_str, fontsize=12, ha='center', va='center', 
         transform=g.fig.transFigure, rotation=90)

g.despine(bottom=True, left=True)

plt.tight_layout()
plt.savefig("th_margin.png", dpi=600)



#plot result of Figure 5
noise_level=[0,0.2,0.4,0.6]
n_list = np.ceil(np.linspace(2**8, 2**14, num=20)).astype(int).tolist()
sim=100
list_sim=[i for i in range(sim)]

the_folder = "CodeR_BTmodel/result"
list_allfile = os.listdir(the_folder)
Regret = np.zeros([len(n_list),len(noise_level)])

for i in range(len(n_list)):
    for j in range(len(noise_level)):
        The_DATA_MARK = '*dCOR_idata_The_sample_size%s_sim_%s_*' %(n_list[i],noise_level[j])
        list_file = []
        for ifile in list_allfile:
            if fnmatch(ifile,The_DATA_MARK):
                list_file.append(ifile)
        len(list_file)
        the_res = []
        for idx_data in range(len(list_file)):
            with open(os.path.join(the_folder,  list_file[idx_data]), 'rb') as file:
                ff = pickle.load(file)
                the_res.append(ff['reg_test'].item())
        Regret[i,j] = np.mean(the_res)
        
loss=Regret
noise_level = [0, 0.2, 0.4, 0.6]
n_list = np.ceil(np.linspace(2**8, 2**14, num=20)).astype(int).tolist()
x =n_list
k_ticks = [8, 10,11, 12,13, 14]  # sparser ticks to avoid clutter
pow_ticks = [2**k for k in k_ticks]
template = "${{2^{{{k}}}}}$"  # braces doubled so .format doesn't eat them
pow_tick_labels = [template.format(k=k) for k in k_ticks]

plt.figure(figsize=(4, 3))

# Use viridis colors (4 evenly spaced colors)
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.15, 0.95, len(noise_level)))  # avoid extremes if desired

for j, nl in enumerate(noise_level):
    plt.plot(
        x, loss[:, j],
        color=colors[j],
        linewidth=2,
        markersize=4,
        label=f"{nl}"
    )

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(True, which='both', linestyle='--', alpha=0.35)
plt.xticks(pow_ticks, pow_tick_labels)
plt.xlabel('sample size', fontsize=10)
plt.ylabel('regret', fontsize=10)
plt.legend(title='noise level', fontsize=10, frameon=False)
plt.tight_layout()
#plt.box(False)
plt.savefig(f"N_BT.png", dpi=600)
plt.show()


noise_level=[0,0.2,0.4,0.6]
n_list = np.ceil(np.linspace(2**8, 2**14, num=20)).astype(int).tolist()
sim=100
list_sim=[i for i in range(sim)]

the_folder = "CodeR_THmodel/result"
list_allfile = os.listdir(the_folder)
Regret = np.zeros([len(n_list),len(noise_level)])

for i in range(len(n_list)):
    for j in range(len(noise_level)):
        The_DATA_MARK = '*dCOR_idata_The_sample_size%s_sim_%s_*' %(n_list[i],noise_level[j])
        list_file = []
        for ifile in list_allfile:
            if fnmatch(ifile,The_DATA_MARK):
                list_file.append(ifile)
        len(list_file)
        the_res = []
        for idx_data in range(len(list_file)):
            with open(os.path.join(the_folder,  list_file[idx_data]), 'rb') as file:
                ff = pickle.load(file)
                the_res.append(ff['reg_test'].item())
        Regret[i,j] = np.mean(the_res)
        
loss=Regret
noise_level = [0, 0.2, 0.4, 0.6]
n_list = np.ceil(np.linspace(2**8, 2**14, num=20)).astype(int).tolist()
x =n_list
k_ticks = [8, 10,11, 12,13, 14]  # sparser ticks to avoid clutter
pow_ticks = [2**k for k in k_ticks]
template = "${{2^{{{k}}}}}$"  # braces doubled so .format doesn't eat them
pow_tick_labels = [template.format(k=k) for k in k_ticks]

plt.figure(figsize=(4, 3))

# Use viridis colors (4 evenly spaced colors)
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.15, 0.95, len(noise_level)))  # avoid extremes if desired

for j, nl in enumerate(noise_level):
    plt.plot(
        x, loss[:, j],
        color=colors[j],
        linewidth=2,
        markersize=4,
        label=f"{nl}"
    )

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.grid(True, which='both', linestyle='--', alpha=0.35)
plt.xticks(pow_ticks, pow_tick_labels)
plt.xlabel('sample size', fontsize=10)
plt.ylabel('regret', fontsize=10)
plt.legend(title='noise level', fontsize=10, frameon=False)
plt.tight_layout()
#plt.box(False)
plt.savefig(f"N_TH.png", dpi=600)
plt.show()