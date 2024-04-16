# EE 559 Project

Kaggle Dataset: https://www.kaggle.com/datasets/ara001/laptop-prices-based-on-its-specifications/data

## Preprocessing and Feature Engineering

### Step 1: Importing Training Dataset and Splitting Features
We imported the data from `laptop_data_train.csv` and split some features into multiple features. 

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Features Before (Total = 11)        ┃ Features After (Total = 16)         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Company: Unique Count = 19          │ Company: Unique Count = 19          │
│ TypeName: Unique Count = 6          │ TypeName: Unique Count = 6          │
│ Inches: Unique Count = 17           │ Inches: Unique Count = 17           │
│ ScreenResolution: Unique Count = 37 │ Display: Unique Count = 10          │
│ Cpu: Unique Count = 100             │ Touchscreen: Unique Count = 2       │
│ Ram: Unique Count = 8               │ Width: Unique Count = 13            │
│ Memory: Unique Count = 33           │ Height: Unique Count = 10           │
│ Gpu: Unique Count = 96              │ MemoryComponent1: Unique Count = 20 │
│ OpSys: Unique Count = 9             │ MemoryComponent2: Unique Count = 6  │
│ Weight: Unique Count = 161          │ CpuModel: Unique Count = 81         │
│ Price: Unique Count = 600           │ ClockSpeed: Unique Count = 25       │
│                                     │ Ram: Unique Count = 8               │
│                                     │ Gpu: Unique Count = 96              │
│                                     │ OpSys: Unique Count = 9             │
│                                     │ Weight: Unique Count = 161          │
│                                     │ Price: Unique Count = 600           │
└─────────────────────────────────────┴─────────────────────────────────────┘

- We split the ScreenResolution category into 4 features: Display (i.e IPS Panel, Full HD, etc.), Touchscreen (i.e. Yes or No), Width, and Height (width and height referring to pixel count).

- We split the CPU category into 2 features: CpuModel (i.e. Intel Core i7, Intel Core i5 7200U) and ClockSpeed (i.e. 2.7 GHz). We checked CPU for extra spaces, and there were none
  
- We split Memory into MemoryCompoennt1 + MemoryComponent2

- Even though GPU has 90 features, we decided against processing it further (besides stripping extra spaces) because each feature of GPU is extremely important in pricing, so there is no need to split the feature further, since they need to be considered all together (i.e. model, series, brand, etc.)


### Step 2: Encoding
We determined how to encode. Our choices were one-hot encoding and label encoding. 
&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — One-Hot Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Preserves uniqueness (each category gets its own binary column, i.e. no ordinal relationship imposed), and it works well with nominal categorical data where there is no intrinsic order. 
&nbsp;&nbsp;&nbsp;&nbsp; Cons: Increases dimensionality, and adds sparse matrices which can consume more memory and computational resources.

&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — Label Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Reduced dimensionality (one column for labels) saves memory and computational resources, preserves order, and simplicity (straightforward to implement)
&nbsp;&nbsp;&nbsp;&nbsp; Cons: May inadvertently introduce ordinal relationships where none exist, thus leading to potentially biased models

The preference of one-hot encoding for nominal categorical data with no inherent order vs. the preference for label encoding for ordinal categorical data with a clear order led us to decide which encoding to use based on the characteristics of each feature. 



All unique singular types of storage:
```
'32GB SSD' '128GB HDD' '1.0TB Hybrid' '64GB SSD' '256GB SSD' '180GB SSD' '32GB HDD' '2TB HDD'
'512GB Flash Storage' '16GB Flash Storage' '64GB Flash Storage' '1TB SSD' '16GB SSD' '500GB HDD'
'32GB Flash Storage' '128GB SSD' 'None' '512GB SSD' '256GB Flash Storage' '1TB HDD' '128GB Flash Storage'
```

Order of pricing based on type of storage: `None < Flash Storage < HDD < Hybrid < SSD`

*However, this dataset is outdated, so the pricing order is not true to the current market. Today, Hybrid storage is no longer in supply so they would be
more expensive to purchase*

Rank of these storage types in (ascending) order of how much they affect the price of a laptop, considering both performance and cost:

1. None:
&nbsp;&nbsp;&nbsp;&nbsp;This refers to no internal storage (e.g., laptops without built-in storage).
&nbsp;&nbsp;&nbsp;&nbsp;It doesn’t directly impact the price since you’ll need to add external storage separately.

2. 16GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Very minimal storage capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Typically found in budget devices.
&nbsp;&nbsp;&nbsp;&nbsp;Low impact on price due to its small size.

3. 32GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Still limited storage.
&nbsp;&nbsp;&nbsp;&nbsp;Budget-friendly but not a significant price factor.

4. 64GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Slightly more storage than 32GB.
&nbsp;&nbsp;&nbsp;&nbsp;Still budget-friendly.

5. 128GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Common in entry-level laptops.
&nbsp;&nbsp;&nbsp;&nbsp;Affects price, but not significantly.

6. 128GB SSD:
&nbsp;&nbsp;&nbsp;&nbsp;Entry-level SSD capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Affects price, especially in budget laptops.

7. 180GB SSD:
&nbsp;&nbsp;&nbsp;&nbsp;Small SSD capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Moderately impacts price.

8. 256GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Decent storage for everyday use.
&nbsp;&nbsp;&nbsp;&nbsp;Affects price, especially in mid-range laptops.

9. 500GB HDD:
&nbsp;&nbsp;&nbsp;&nbsp;HDD with good capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Budget-friendly but slower performance.

10. 512GB Flash Storage:
&nbsp;&nbsp;&nbsp;&nbsp;Larger SSD capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Noticeable impact on price.

11. 256GB SSD:
&nbsp;&nbsp;&nbsp;&nbsp;Balanced SSD capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Common and moderately affects price.

12. 1TB HDD:
&nbsp;&nbsp;&nbsp;&nbsp;HDD with ample storage.
&nbsp;&nbsp;&nbsp;&nbsp;Common in mid-range laptops.

13. 1.0TB Hybrid (SSHD):
&nbsp;&nbsp;&nbsp;&nbsp;Combines HDD and SSD.
&nbsp;&nbsp;&nbsp;&nbsp;Moderately impacts price.

14.  1TB SSD:
&nbsp;&nbsp;&nbsp;&nbsp;Large SSD capacity.
&nbsp;&nbsp;&nbsp;&nbsp;Significantly affects the price due to SSD cost per gigabyte1.

15. 2TB HDD:
&nbsp;&nbsp;&nbsp;&nbsp;High-capacity HDD.
&nbsp;&nbsp;&nbsp;&nbsp;Adds to the price but still reasonable.


Feature 8, x8: *(Total: 96)*

```
 ['AMD FirePro W4190M ' 'AMD FirePro W5130M' 'AMD R17M-M1-70'
 'AMD R4 Graphics' 'AMD Radeon 520' 'AMD Radeon 530' 'AMD Radeon 540'
 'AMD Radeon Pro 455' 'AMD Radeon Pro 555' 'AMD Radeon Pro 560'
 'AMD Radeon R2' 'AMD Radeon R2 Graphics' 'AMD Radeon R3' 'AMD Radeon R4'
 'AMD Radeon R4 Graphics' 'AMD Radeon R5' 'AMD Radeon R5 430'
 'AMD Radeon R5 520' 'AMD Radeon R5 M420' 'AMD Radeon R5 M420X'
 'AMD Radeon R5 M430' 'AMD Radeon R7' 'AMD Radeon R7 Graphics'
 'AMD Radeon R7 M440' 'AMD Radeon R7 M445' 'AMD Radeon R7 M460'
 'AMD Radeon R7 M465' 'AMD Radeon RX 540' 'AMD Radeon RX 550'
 'AMD Radeon RX 560' 'AMD Radeon RX 580' 'Intel Graphics 620'
 'Intel HD Graphics' 'Intel HD Graphics 400' 'Intel HD Graphics 405'
 'Intel HD Graphics 500' 'Intel HD Graphics 505' 'Intel HD Graphics 510'
 'Intel HD Graphics 515' 'Intel HD Graphics 520' 'Intel HD Graphics 530'
 'Intel HD Graphics 5300' 'Intel HD Graphics 540' 'Intel HD Graphics 6000'
 'Intel HD Graphics 615' 'Intel HD Graphics 620' 'Intel HD Graphics 630'
 'Intel Iris Graphics 540' 'Intel Iris Graphics 550'
 'Intel Iris Plus Graphics 640' 'Intel Iris Plus Graphics 650'
 'Intel Iris Pro Graphics' 'Intel UHD Graphics 620' 'Nvidia GTX 980 SLI'
 'Nvidia GeForce 150MX' 'Nvidia GeForce 920' 'Nvidia GeForce 920M'
 'Nvidia GeForce 920MX' 'Nvidia GeForce 920MX ' 'Nvidia GeForce 930M'
 'Nvidia GeForce 930MX' 'Nvidia GeForce 930MX ' 'Nvidia GeForce 940M'
 'Nvidia GeForce 940MX' 'Nvidia GeForce GT 940MX'
 'Nvidia GeForce GTX 1050' 'Nvidia GeForce GTX 1050 Ti'
 'Nvidia GeForce GTX 1050M' 'Nvidia GeForce GTX 1050Ti'
 'Nvidia GeForce GTX 1060' 'Nvidia GeForce GTX 1070'
 'Nvidia GeForce GTX 1070M' 'Nvidia GeForce GTX 1080'
 'Nvidia GeForce GTX 930MX' 'Nvidia GeForce GTX 940M'
 'Nvidia GeForce GTX 940MX' 'Nvidia GeForce GTX 950M'
 'Nvidia GeForce GTX 960' 'Nvidia GeForce GTX 960<U+039C>'
 'Nvidia GeForce GTX 960M' 'Nvidia GeForce GTX 965M'
 'Nvidia GeForce GTX 970M' 'Nvidia GeForce GTX 980M'
 'Nvidia GeForce GTX1050 Ti' 'Nvidia GeForce GTX1060'
 'Nvidia GeForce GTX1080' 'Nvidia GeForce MX130' 'Nvidia GeForce MX150'
 'Nvidia Quadro M1000M' 'Nvidia Quadro M1200' 'Nvidia Quadro M2000M'
 'Nvidia Quadro M2200' 'Nvidia Quadro M2200M' 'Nvidia Quadro M520M'
 'Nvidia Quadro M620' 'Nvidia Quadro M620M']
```

Feature 9, x9: *(Total: 9)*
```['Android' 'Chrome OS' 'Linux' 'Mac OS X' 'No OS' 'Windows 10' 'Windows 10 S' 'Windows 7' 'macOS']```
