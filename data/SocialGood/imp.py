import pandas as pd

# 1. File paths
input_path = './data/SocialGood/Unadj_UnemploymentRate_ALL_processed.csv'
output_path = './data/SocialGood/Unadj_UnemploymentRate_ALL_processed_fixed.csv'  # New file name

# 2. Read data
df = pd.read_csv(input_path)

# 3. Fill NaN in 'OT' column only (Forward fill first, then Backward fill)
df['OT'] = df['OT'].ffill().bfill()
df['Final_Search_4'] = df['Final_Search_4'].fillna("Missing")

# 4. Save to a new file (without overwriting)
df.to_csv(output_path, index=False)

# 5. Check for remaining NaNs in OT
nan_count = df['OT'].isnull().sum()
print(f"Saved to: {output_path}")
print(f"Remaining NaNs in OT: {nan_count}")

nan_count = df['Final_Search_4'].isnull().sum()
print(f"Remaining NaNs in Final_Search_4: {nan_count}")