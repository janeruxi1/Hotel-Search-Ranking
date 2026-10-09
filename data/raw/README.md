# Raw data (not committed)

Source: Kaggle competition "Personalize Expedia Hotel Searches – ICDM 2013"
(https://www.kaggle.com/competitions/expedia-personalized-sort). The data belong to
Expedia and are used under the competition rules; they may not be redistributed, so
nothing in this folder is committed except this file.

## How to get it

1. Sign in to Kaggle and accept the rules at
   https://www.kaggle.com/competitions/expedia-personalized-sort/rules
2. On the Data tab, click **Download All** and save `expedia-personalized-sort.zip` here.
3. Extract it, then extract the `data.zip` inside it, so that `train.csv` sits in this folder.
   `data.zip` uses **Deflate64** compression, which Python's `zipfile` cannot read. Use
   Windows Explorer ("Extract All"), 7-Zip, or `unzip`.
4. Run `python scripts/prepare_data.py` from the project root.

`test.csv` has no clicks or bookings and is not used.

Expected `train.csv`: 9,917,530 rows, 54 columns, 399,344 searches, 136,886 hotels,
2012-11-01 to 2013-06-30. `prepare_data.py` checks these counts.
