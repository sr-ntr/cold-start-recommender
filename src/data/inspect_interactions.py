import pandas as pd

FILE_PATH = "data/raw/Electronics.csv.gz"

# First inspect the schema using only a small sample.
sample = pd.read_csv(FILE_PATH, nrows=5)

print("Columns:")
print(sample.columns.tolist())

print("\nSample:")
print(sample)

print("\nData types:")
print(sample.dtypes)

# Aggregate statistics without loading the whole dataset.
rating_counts = {}
user_ids = set()
item_ids = set()

print("\nScanning dataset in chunks...")

for chunk in pd.read_csv(FILE_PATH, chunksize=500_000):
    # Rating distribution
    counts = chunk["rating"].value_counts()
    for rating, count in counts.items():
        rating_counts[rating] = rating_counts.get(rating, 0) + count

    # Unique IDs
    user_ids.update(chunk["user_id"].unique())
    item_ids.update(chunk["parent_asin"].unique())

print("\nRating distribution:")
for rating, count in sorted(rating_counts.items()):
    print(f"{rating}: {count:,}")

print(f"\nUnique users: {len(user_ids):,}")
print(f"Unique items: {len(item_ids):,}")

# ---------------------------------------------------------
# Item interaction distribution
# ---------------------------------------------------------

item_counts = {}

print("\nCounting interactions per item...")

for chunk in pd.read_csv(FILE_PATH, chunksize=500_000):
    counts = chunk["parent_asin"].value_counts()

    for item, count in counts.items():
        item_counts[item] = item_counts.get(item, 0) + count

item_counts_series = pd.Series(item_counts)

print("\nItem interaction count statistics:")
print(item_counts_series.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

print("\nItems by interaction count:")
print(f"Exactly 1:      {(item_counts_series == 1).sum():,}")
print(
    f"2-5:            {((item_counts_series >= 2) & (item_counts_series <= 5)).sum():,}"
)
print(
    f"6-20:           {((item_counts_series >= 6) & (item_counts_series <= 20)).sum():,}"
)
print(f"21+:            {(item_counts_series >= 21).sum():,}")

# ---------------------------------------------------------
# Timestamp range
# ---------------------------------------------------------

min_timestamp = None
max_timestamp = None

for chunk in pd.read_csv(FILE_PATH, usecols=["timestamp"], chunksize=500_000):
    chunk_min = chunk["timestamp"].min()
    chunk_max = chunk["timestamp"].max()

    if min_timestamp is None or chunk_min < min_timestamp:
        min_timestamp = chunk_min

    if max_timestamp is None or chunk_max > max_timestamp:
        max_timestamp = chunk_max

print("\nTimestamp range:")
print("Minimum:", min_timestamp)
print("Maximum:", max_timestamp)

# ---------------------------------------------------------
# Interaction distribution over time
# ---------------------------------------------------------

year_counts = {}

for chunk in pd.read_csv(FILE_PATH, usecols=["timestamp"], chunksize=500_000):
    dates = pd.to_datetime(chunk["timestamp"], unit="ms")
    counts = dates.dt.year.value_counts()

    for year, count in counts.items():
        year_counts[year] = year_counts.get(year, 0) + count

print("\nInteractions by year:")

for year, count in sorted(year_counts.items()):
    print(f"{year}: {count:,}")
