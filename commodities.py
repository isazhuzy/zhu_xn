import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(
    "Copy of wind品种指数数据(1).xlsx",
    sheet_name="Sheet1",
    header=3
)

df.dropna(axis=1, how="all", inplace=True) #dropping empty columns
df = df.iloc[1:].copy() #removing code row
df["日期"] = pd.to_datetime(df["日期"])
for col in df.columns[1:]: #prices to numbers
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.set_index("日期", inplace=True)
ret_1d = df.pct_change() #computing returns
trend_sharpe = ( #annualised trend sharpe
    np.sqrt(252)
    * ret_1d.mean()
    / ret_1d.std()
)
trend_sharpe = trend_sharpe.sort_values()

#plotting
plt.rcParams["font.sans-serif"] = ["PingFang HK"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(12,10))

trend_sharpe.plot(
    kind="barh",
    ax=ax
)

for i, v in enumerate(trend_sharpe):
    ax.text(v, i, f"{v:.2f}", va="center")

plt.title("趋势流畅度对比")
plt.xlabel("Trend Sharpe")

plt.tight_layout()
plt.savefig(
    "trend_sharpe.png"
)
plt.show()