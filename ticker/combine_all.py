import pandas as pd

df_ic = pd.read_csv("IC_20230104_20230304.csv")
df_if = pd.read_csv("IF_20230104_20230304.csv")
df_ih = pd.read_csv("IH_20230104_20230304.csv")
df_im = pd.read_csv("IM_20230104_20230304.csv")

df_all = pd.concat(
    [df_ic, df_if, df_ih, df_im],
    ignore_index=True
)

print(df_all.shape)

df_all.to_csv(
    "IC_IF_IH_IM_20230104_20230304.csv",
    index=False
)