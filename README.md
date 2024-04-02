# EE 559 Project

Kaggle Dataset: https://www.kaggle.com/datasets/ara001/laptop-prices-based-on-its-specifications/data

## Preprocessing and Feature Engineering

### Step 1: Importing Training Dataset and Encoding 
We imported the data from `laptop_data_train.csv` and split some features into multiple features. 

We split the CPU category into 3 features: Core Line Name (i.e. Intel Core i7), Core Model/Generation (i.e. 6820HK), Processor Clock Speed (i.e. 2.7 GHz).


### Step 2: Splitting features into multiple features
We determined how to encode. Our choices were one-hot encoding and label encoding. 
&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — One-Hot Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Preserves uniqueness (each category gets its own binary column, i.e. no ordinal relationship imposed), and it works well with nominal categorical data where there is no intrinsic order. 
&nbsp;&nbsp;&nbsp;&nbsp; Cons: Increases dimensionality, and adds sparse matrices which can consume more memory and computational resources.

&nbsp;&nbsp;&nbsp;&nbsp; **Option 1 — Label Encoding**
&nbsp;&nbsp;&nbsp;&nbsp; Pros: Reduced dimensionality (one column for labels) saves memory and computational resources, preserves order, and simplicity (straightforward to implement)
&nbsp;&nbsp;&nbsp;&nbsp; Cons: May inadvertently introduce ordinal relationships where none exist, thus leading to potentially biased models

The preference of one-hot encoding for nominal categorical data with no inherent order vs. the preference for label encoding for ordinal categorical data with a clear order led us to decide which encoding to use based on the characteristics of each feature. 

Feature 1, x1 – Company: (Total: 19)
   
```['Acer' 'Apple' 'Asus' 'Chuwi' 'Dell' 'Fujitsu' 'Google' 'HP' 'Huawei' 'LG' 'Lenovo' 'MSI' 'Mediacom' 'Microsoft' 'Razer' 'Samsung' 'Toshiba' 'Vero' 'Xiaomi']```

Feature 2, x2 – Company: *(Total: 6)*

```['2 in 1 Convertible' 'Gaming' 'Netbook' 'Notebook' 'Ultrabook' 'Workstation']```

Feature 3, x3 – : *(Total: 17)*

```['10.1' '11.6' '12.0' '12.3' '12.5' '13.0' '13.3' '13.5' '13.9' '14.0' '14.1' '15.0' '15.4' '15.6' '17.0' '17.3' '18.4']```

Feature 4, x4: *(Total: 37)*

```['1366x768' '1440x900' '1600x900' '1920x1080' '2560x1440' '4K Ultra HD / Touchscreen 3840x2160' '4K Ultra HD 3840x2160' 'Full HD / Touchscreen 1920x1080' 'Full HD 1920x1080' 'IPS Panel 1366x768' 'IPS Panel 2560x1440' 'IPS Panel 4K Ultra HD / Touchscreen 3840x2160' 'IPS Panel 4K Ultra HD 3840x2160' 'IPS Panel Full HD / Touchscreen 1920x1080' 'IPS Panel Full HD 1366x768' 'IPS Panel Full HD 1920x1080' 'IPS Panel Full HD 1920x1200' 'IPS Panel Full HD 2160x1440' 'IPS Panel Full HD 2560x1440' 'IPS Panel Quad HD+ / Touchscreen 3200x1800' 'IPS Panel Quad HD+ 2560x1440' 'IPS Panel Quad HD+ 3200x1800' 'IPS Panel Retina Display 2304x1440' 'IPS Panel Retina Display 2560x1600' 'IPS Panel Retina Display 2736x1824' 'IPS Panel Retina Display 2880x1800' 'IPS Panel Touchscreen / 4K Ultra HD 3840x2160' 'IPS Panel Touchscreen 1366x768' 'IPS Panel Touchscreen 1920x1200' 'IPS Panel Touchscreen 2560x1440' 'Quad HD+ / Touchscreen 3200x1800' 'Touchscreen / Full HD 1920x1080' 'Touchscreen / Quad HD+ 3200x1800' 'Touchscreen 1366x768' 'Touchscreen 2256x1504' 'Touchscreen 2400x1600' 'Touchscreen 2560x1440']```

Feature 5, x5: *(Total: 100)*

```
['AMD A10-Series 9600P 2.4GHz' 'AMD A10-Series 9620P 2.5GHz'
 'AMD A10-Series A10-9620P 2.5GHz' 'AMD A12-Series 9700P 2.5GHz'
 'AMD A12-Series 9720P 2.7GHz' 'AMD A12-Series 9720P 3.6GHz'
 'AMD A4-Series 7210 2.2GHz' 'AMD A6-Series 7310 2GHz'
 'AMD A6-Series 9220 2.5GHz' 'AMD A6-Series 9220 2.9GHz'
 'AMD A6-Series A6-9220 2.5GHz' 'AMD A8-Series 7410 2.2GHz'
 'AMD A9-Series 9420 2.9GHz' 'AMD A9-Series 9420 3GHz'
 'AMD A9-Series A9-9420 3GHz' 'AMD E-Series 6110 1.5GHz'
 'AMD E-Series 7110 1.8GHz' 'AMD E-Series 9000e 1.5GHz'
 'AMD E-Series E2-6110 1.5GHz' 'AMD E-Series E2-9000 2.2GHz'
 'AMD E-Series E2-9000e 1.5GHz' 'AMD FX 9830P 3GHz'
 'AMD Ryzen 1600 3.2GHz' 'AMD Ryzen 1700 3GHz' 'Intel Atom Z8350 1.92GHz'
 'Intel Atom x5-Z8300 1.44GHz' 'Intel Atom x5-Z8350 1.44GHz'
 'Intel Atom x5-Z8550 1.44GHz' 'Intel Celeron Dual Core 3205U 1.5GHz'
 'Intel Celeron Dual Core 3855U 1.6GHz'
 'Intel Celeron Dual Core N3050 1.6GHz'
 'Intel Celeron Dual Core N3060 1.60GHz'
 'Intel Celeron Dual Core N3060 1.6GHz'
 'Intel Celeron Dual Core N3350 1.1GHz'
 'Intel Celeron Dual Core N3350 2.0GHz'
 'Intel Celeron Quad Core N3160 1.6GHz'
 'Intel Celeron Quad Core N3450 1.1GHz'
 'Intel Celeron Quad Core N3710 1.6GHz' 'Intel Core M 1.2GHz'
 'Intel Core M 6Y75 1.2GHz' 'Intel Core M 7Y30 1.0GHz'
 'Intel Core M M3-6Y30 0.9GHz' 'Intel Core M m3 1.2GHz'
 'Intel Core M m3-7Y30 2.2GHz' 'Intel Core i3 6006U 2.0GHz'
 'Intel Core i3 6006U 2.2GHz' 'Intel Core i3 6006U 2GHz'
 'Intel Core i3 6100U 2.1GHz' 'Intel Core i3 6100U 2.3GHz'
 'Intel Core i3 7100U 2.4GHz' 'Intel Core i3 7130U 2.7GHz'
 'Intel Core i5 1.3GHz' 'Intel Core i5 1.6GHz' 'Intel Core i5 1.8GHz'
 'Intel Core i5 2.0GHz' 'Intel Core i5 2.3GHz' 'Intel Core i5 2.9GHz'
 'Intel Core i5 3.1GHz' 'Intel Core i5 6200U 2.3GHz'
 'Intel Core i5 6260U 1.8GHz' 'Intel Core i5 6300HQ 2.3GHz'
 'Intel Core i5 6300U 2.4GHz' 'Intel Core i5 6440HQ 2.6GHz'
 'Intel Core i5 7200U 2.5GHz' 'Intel Core i5 7300HQ 2.5GHz'
 'Intel Core i5 7300U 2.6GHz' 'Intel Core i5 7440HQ 2.8GHz'
 'Intel Core i5 7500U 2.7GHz' 'Intel Core i5 7Y54 1.2GHz'
 'Intel Core i5 7Y57 1.2GHz' 'Intel Core i5 8250U 1.6GHz'
 'Intel Core i7 2.2GHz' 'Intel Core i7 2.7GHz' 'Intel Core i7 2.8GHz'
 'Intel Core i7 2.9GHz' 'Intel Core i7 6500U 2.5GHz'
 'Intel Core i7 6600U 2.6GHz' 'Intel Core i7 6700HQ 2.6GHz'
 'Intel Core i7 6820HK 2.7GHz' 'Intel Core i7 6820HQ 2.7GHz'
 'Intel Core i7 6920HQ 2.9GHz' 'Intel Core i7 7500U 2.5GHz'
 'Intel Core i7 7500U 2.7GHz' 'Intel Core i7 7560U 2.4GHz'
 'Intel Core i7 7600U 2.8GHz' 'Intel Core i7 7660U 2.5GHz'
 'Intel Core i7 7700HQ 2.7GHz' 'Intel Core i7 7700HQ 2.8GHz'
 'Intel Core i7 7820HK 2.9GHz' 'Intel Core i7 7820HQ 2.9GHz'
 'Intel Core i7 7Y75 1.3GHz' 'Intel Core i7 8550U 1.8GHz'
 'Intel Core i7 8650U 1.9GHz' 'Intel Pentium Dual Core 4405U 2.1GHz'
 'Intel Pentium Dual Core N4200 1.1GHz'
 'Intel Pentium Quad Core N3710 1.6GHz'
 'Intel Pentium Quad Core N4200 1.1GHz' 'Intel Xeon E3-1505M V6 3GHz'
 'Intel Xeon E3-1535M v5 2.9GHz' 'Intel Xeon E3-1535M v6 3.1GHz']
```

Feature 6, x6: *(Total: 8)*

```['12GB' '16GB' '24GB' '2GB' '32GB' '4GB' '6GB' '8GB']```

Feature 7, x7: *(Total: 33)*

```
 ['1.0TB Hybrid' '128GB Flash Storage' '128GB HDD' '128GB SSD'
 '128GB SSD +  1TB HDD' '128GB SSD +  2TB HDD' '16GB Flash Storage'
 '16GB SSD' '180GB SSD' '1TB HDD' '1TB HDD +  1TB HDD' '1TB SSD'
 '1TB SSD +  1TB HDD' '256GB Flash Storage' '256GB SSD'
 '256GB SSD +  1TB HDD' '256GB SSD +  256GB SSD' '256GB SSD +  2TB HDD'
 '256GB SSD +  500GB HDD' '2TB HDD' '32GB Flash Storage' '32GB HDD'
 '32GB SSD' '500GB HDD' '512GB Flash Storage' '512GB SSD'
 '512GB SSD +  1TB HDD' '512GB SSD +  256GB SSD' '512GB SSD +  2TB HDD'
 '512GB SSD +  512GB SSD' '64GB Flash Storage'
 '64GB Flash Storage +  1TB HDD' '64GB SSD']
```

All unique singular types of storage:
```
'32GB SSD' '128GB HDD' '1.0TB Hybrid' '64GB SSD' '256GB SSD' '180GB SSD' '32GB HDD' '2TB HDD'
'512GB Flash Storage' '16GB Flash Storage' '64GB Flash Storage' '1TB SSD' '16GB SSD' '500GB HDD'
'32GB Flash Storage' '128GB SSD' 'None' '512GB SSD' '256GB Flash Storage' '1TB HDD' '128GB Flash Storage'
```

Order of pricing based on type of storage: `None < Flash Storage < HDD < Hybrid < SSD`

*However, this dataset is outdated, so the pricing order is not true to the current market. Today, Hybrid storage is no longer in supply so they would be
more expensive to purchase*

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
