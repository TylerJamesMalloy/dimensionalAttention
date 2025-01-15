import os 
import pandas as pd 
import scipy.io as sio
import seaborn as sns 
import matplotlib.pyplot as plt 

directory = 'files'
 
# iterate over files in
# that directory
ldf = pd.DataFrame([])
mdf = pd.DataFrame([])
tdf = pd.DataFrame([])

for directory in ['./Complete', './Partial']:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            file_name, file_extension = os.path.splitext(f)
            if(file_extension == '.mat'):
                participant = file_name.split('_')[0].split("\\")[1]
                mat_data = sio.loadmat(f)
                # 'data_Learning', 'data_Mean', 'data_Test'
                if('Learn' in file_name):
                    data = mat_data['data_Learning']
                    d = pd.DataFrame(data)
                    d['Type'] = directory.split('/')[1]
                    d['Participant'] = participant
                    d.rename(columns={14: "Reward"}, inplace=True)
                    d.rename(columns={2: "Timestep"}, inplace=True)
                    ldf = pd.concat([ldf, d])


ldf.to_pickle("./Learn.pkl")
grp = ldf.groupby([4, "Type", "Participant", "Timestep"], as_index=False).mean()

c = ldf[ldf['Type'] == "Complete"]
sns.lineplot(data=c, x='Timestep', y=14, hue=4)
plt.show()
#print(ldf[4].unique())

c = ldf[ldf['Type'] == 'Complete']
p = ldf[ldf['Type'] == 'Partial']

#print(c['Reward'].mean())
#print(p['Reward'].mean())

"""
import os 
import pandas as pd 
import scipy.io as sio

directory = 'files'
 
# iterate over files in
# that directory
ldf = pd.DataFrame([])
mdf = pd.DataFrame([])
tdf = pd.DataFrame([])

for directory in ['./Complete', './Partial']:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            file_name, file_extension = os.path.splitext(f)
            print(file_name)
            assert(False)
            if(file_extension == '.mat'):
                #print(f)
                mat_data = sio.loadmat(f)
                # 'data_Learning', 'data_Mean', 'data_Test'
                if('Learn' in file_name):
                    data = mat_data['data_Learning']
                    d = pd.DataFrame(data)
                    d['Type'] = directory
                    ldf = pd.concat([ldf, d])
                if('Mean' in file_name):
                    data = mat_data['data_Mean']
                    d = pd.DataFrame(data)
                    d['Type'] = directory
                    mdf = pd.concat([ldf, d])
                if('Test' in file_name):
                    data = mat_data['data_Test']
                    d = pd.DataFrame(data)
                    d['Type'] = directory
                    tdf = pd.concat([ldf, d])
tdf.to_pickle("./Test.pkl")
mdf.to_pickle("./Mean.pkl")
ldf.to_pickle("./Learn.pkl")

"""