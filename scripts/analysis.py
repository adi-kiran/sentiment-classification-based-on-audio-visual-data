#%%
import pandas as pd


# %%
df = pd.read_csv("All-emotions.csv")
cols = df.columns
print(cols)

# %%
truth_value = df["emotion"]
tru = set(truth_value)
print(tru)

# %%
print(df.shape)

# %%
