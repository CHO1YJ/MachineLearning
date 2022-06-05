import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inlinesns.set(font_scale=2)

array = [[5, 0, 0, 0], [0, 10, 0, 0], [0, 0, 15, 0], [0, 0, 0, 5]]
df_cm = pd.DataFrame(array, index = [i for i in "ABCD"], columns=[i for i in 'ABCD'])
plt.figure(figsize=(10, 7))
plt.title("CM")
sns.heatmap(df_cm, annot=True)