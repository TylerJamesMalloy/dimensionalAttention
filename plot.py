import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 

df = pd.read_pickle("./Results/Simulations.pkl")
fig = plt.figure(figsize=(16, 4))
# Top Row
ax1 = plt.subplot2grid((1, 6), (0, 0), colspan=1)
ax2 = plt.subplot2grid((1, 6), (0, 1), colspan=1)
ax3 = plt.subplot2grid((1, 6), (0, 2), colspan=1)
ax4 = plt.subplot2grid((1, 6), (0, 3), colspan=1)
ax5 = plt.subplot2grid((1, 6), (0, 4), colspan=1)
ax6 = plt.subplot2grid((1, 6), (0, 5), colspan=1)

iblPalette = sns.color_palette(palette='tab10')[0:2]
iblPalette.reverse()
frlPalette = sns.color_palette(palette='tab10')[3:5]
palette = iblPalette + frlPalette

model = "IBL"
ibl = df[(df['Name'] == 'IBL') | (df['Name'] == 'WIBL')]
im=df[df['Feedback'] == "Immediate"]
sns.lineplot(im[im["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax1, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(im[im["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax2, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
ax1.set_title("Intradimensional", fontsize=16)
ax2.set_title("Extradimensional", fontsize=16)

cl=df[df['Feedback'] == "Clustered"]
sns.lineplot(cl[cl["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax3, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(cl[cl["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax4, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
ax3.set_title("Intradimensional", fontsize=16)
ax4.set_title("Extradimensional", fontsize=16)

ad=df[df['Feedback'] == "Additional"]
sns.lineplot(ad[ad["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax5, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(ad[ad["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax6, hue="Name", palette=iblPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
ax5.set_title("Intradimensional", fontsize=16)
ax6.set_title("Extradimensional", fontsize=16)

model = "FRL"
frl = df[(df['Name'] == 'FRL') | (df['Name'] == 'WFRL')]
im = frl[frl["Feedback"] == "Immediate"]
sns.lineplot(im[im["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax1, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(im[im["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax2, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))

cl = frl[frl["Feedback"] == "Clustered"]
sns.lineplot(cl[cl["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax3, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(cl[cl["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax4, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))

ad = frl[frl["Feedback"] == "Additional"]
sns.lineplot(ad[ad["IntraDimensional"] == 1], x="Timestep", y="Correct", ax=ax5, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))
sns.lineplot(ad[ad["ExtraDimensional"] == 1], x="Timestep", y="Correct", ax=ax6, hue="Name", palette=frlPalette, hue_order=['W' + model, model], errorbar=('ci', 68))


ax1.set_ylabel("Probability of Correct Choice", fontsize=14)
ax1.legend(fontsize=14) 

for ax in [ax2, ax3, ax4, ax5, ax6]:
       ax.set_ylabel("")
       ax.set_yticks([])
       ax.set_yticks([], minor=True)
       ax.get_legend().remove()

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
       ax.set_ylim(0.25, 1)
       ax.set_xlabel("Timestep", fontsize=14)

# Get the legend handles and labels
handles, labels = ax1.get_legend_handles_labels()

# Loop through the handles and set the linewidth
for handle in handles:
    handle.set_linewidth(5)  # Set the desired line width

ax1.legend(handles, labels)
          
plt.subplots_adjust(left=0.05,bottom=0.12, right=0.995, top=0.86, wspace=0, hspace=0)
fig.text(0.2, 0.945, 'Immediate Feedback', ha='center', fontsize=20)
fig.text(0.525, 0.945, 'Delayed Feedback', ha='center', fontsize=20)
fig.text(0.85, 0.945 , 'Counterfactual Feedback', ha='center', fontsize=20)

plt.show()