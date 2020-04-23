#%%
import pandas as pd

# %%
original = pd.read_csv("merged.csv")
df = pd.DataFrame()

# %%
rslt_df = original.loc[original['emotion'] == 3]

# %%
rslt_df

# %%
rslt_df2 = original.loc[original['emotion'] == 4]

# %%
rslt_df2

# %%
df = rslt_df.append(rslt_df2, ignore_index=True)

# %%
df

# %%
df2 = df[df['12A'] != 0.0]

# %%
df2

# %%
df2.loc[df['emotion'] == 3, 'emotion'] = 1
df2.loc[df['emotion'] == 4, 'emotion'] = 0

# %%
df2

# %%
# df2 = df2.sample(frac = 1)
# Shuffling so that the classifications are not in order
# %%
# df2

# %%
df2.to_csv("unshuffled_binary.csv")

# %%
