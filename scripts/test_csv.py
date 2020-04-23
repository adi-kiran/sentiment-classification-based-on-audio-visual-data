#%%
import pandas as pd

# %%
df = pd.read_csv("Small.csv")
df.shape

# %%
df.loc[df['emotion'] == 3, 'emotion'] = 1 # 1 is happy
df.loc[df['emotion'] == 4, 'emotion'] = 0 # 0 is sad
truth = df["emotion"]

# %%
set_truth = set(truth)
print(set_truth)

# %%
