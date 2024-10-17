import pandas as pd
from glob import glob
out_df_list = []
for out_csv in glob("results/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
out_df = pd.concat(out_df_list, ignore_index=True)
out_df.to_csv("final_result.csv", index=False)