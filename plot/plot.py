import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,
})

# load data
# csv_name = "bfmax.csv"
csv_name = "hybrid.csv"
df = pd.read_csv(csv_name)

# scatter plot
plt.figure()
sc = plt.scatter(
    df["p"],
    df["dfr"],
    c=df["cc"],
    cmap="viridis"
)

plt.yscale("log")
plt.title(csv_name)
plt.xlabel("p")
plt.ylabel("dfr")
plt.colorbar(sc, label="cc")
plt.show()