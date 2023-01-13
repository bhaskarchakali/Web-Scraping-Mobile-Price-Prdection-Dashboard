
#Importing necessary libraries
import numpy as np
import pandas as pd
df=pd.read_csv('C:/Users/Home/Desktop/Interview project/Fmobiles_clean.csv')
df.info()
df.columns
# ## Exploring the dataset
df.drop(axis=1,columns=['Unnamed: 0'],inplace=True)
df.head()
df['BRAND'].unique()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df.groupby(["BRAND"]).describe().T
# #### BRAND VS MRP
df.groupby(["BRAND"]).describe()["MRP"].T

                         #Observations
# - It is observed that the highest price offerning mobile Apple.
# - we observe that mean and median for Brands 1,BLU,BlackBeaaaar,BlackZone,DETEL,HTC,Hicell,Huawai,ITEL,Infocus, is same.
# - we observe that mean of oneplus brand is nearer to median by 1600.

# #### BRAND VS OFFER_PRICE
df.groupby(["BRAND"]).describe()["OFFER_PRICE"].T
# #### Observations
# - It is observed that the minimum offer price offered by Apple brand is 25299.
# - It is also observed that mean and median are same for Brands 1,BLU,BLACKBEAR,BlackZone,DEtel,HTC,HUawei,Infocus,M,amd for only mobile offering Brands.

# #### BRAND VS MEMORY
df.groupby(["BRAND"]).describe()["MEMORY"].T

                         #Observations: 
# - It is observed that the Apple is offering min of 32 GB and Max. of 512 GB for mobile.
# - Asus is offering min of 32 GB and Max of 128 GB
# - Google is offering min of 32 GB and Max of 128 GB
# - Honor is offering min of 32 GB and Max of 128 GB
# - Motorolo is offering min of 32 GB and Max of 128 GB
# - Oneplus  is offering min of 64 GB and Max of 256 GB
# - It is observed that Real me is offering 256 GB memory for its mobile.

#BRAND VS STAR_RATING
df.groupby(["BRAND"]).describe()["STAR_RATING"].T

                        #Observations
# - The mean and median are nearer for Brand Apple.
# - The mean and median are same for Brands like  1,BLU,BlackBear,Coolpad,Forme,GLx,..
# - The mean and median are nearer for GOogle,Honor,Motorola,Nokia,Oppo,Oneplus
# - The Brand Realme is shown with max of 4.6 star rating.

# #### BRAND VS WARRANTY
df.groupby(["BRAND"]).describe()["WARRANTY"].T
# #### Observations
# - It is observed that max Brands offering 1 year warranty.
# - It is also seen that mean and median are same for max of the brands.

# #### BRAND VS PRODUCT_NAME VS BATTERY
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["BATTERY_C"].T

                        #Observations:
# - It is observed that BRAND Apple is not mentioned about Battery Cappacity
# - Asus is offering 3300 to 4000 caapacity batteries for their mobiles.
# - It is observed that Realmi is offering upto 4000 mah Battery capacity.
# - It is observed that Redmi is offering upto 5000 mah Battery capacity.

# #### BRAND VS PRODUCTNAME VS SCREEN_SIZE
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["SCREEN_SIZE"].T

                          #Observations:
# - It is observed that Brand Apple is offering 4.7 inch screen for lowest mobile and 6.5 inch screen for their highest spec mobile.
# - Google is offering 5.5 inch for lower price mobile and 6.3 inch for higher price mobile.
# - It is observed that Brand oppo is offering 6.5 inch for their most of mobiles
# - It is observed that poco is offering 6.67 inch screen for their mobile.
# - samsung is offering upto 6.7 inch for their mobiles.
# - Realme x2 pro is offering 6.5 inch screen.

# #### BRAND VS PRODUCT NAME VS MEMORY
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["MEMORY"].T

                      #Observations
# - It is observed that from the brand Apple prouct iphone 11 pro and 11 pro max are offering from the base 64 GB to thr top 512 GB.
# - From the other Brands like samsung,Realme ,Poco, oppo,vivo are offering max up to 256 GB for their products like Realme x2 pro,poco x2,Samsung s20-s20+-s20ultra,and others.
# - It is observed that Realme,poco also offering 256 GB for lowest prices.

# #### BRAND VS PRODUCT_NAME VS MRP
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["MRP"].T

                           #Observations
# - It is observed that Apple iphone 11pro max is at the top with Mrp-150800,
# - It is also observed that Apple is Brand with highest listed MRP.
# - The next followed by samsung,oppo.
# - Realme x2 pro with 256 GB is offering at 35999 MRP
# - The Other Brands offering mobiles at low MRP

# #### BRAND VS PRODUCT_NAME VS OFFER_Price
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["OFFER_PRICE"].T

                              #Observations
# - We observe that iphone 11pro-max is at same as MRp and the products from Apple are not offered prices.
# - Here the offer price are same as MRP.

# ### BRAND VS PRODUCT_NAME VS DISCOUNT VS MRP
df.groupby(["BRAND","PRODUCT_NAME","DISCOUNT"]).describe()["MRP"]

#Observations
# - We can observe that 0% discount is offered on Apple products
# - We also observe that the products from Apple like iphone 6s at 15% discount and 7 at 47% discoun ant rest 6,7,8 are offered less than 10@ discount.
# - we also observe that the the Brand ASUS is offering more than 30% discount to all of its products.
# - We observe that google offering upto 30% discount to its products.
# - We observe that Honor is offereing above 40% and for some products 50% on its products.
# - We also observe that micromax is offering greater than 50% discount on its products.
# - The other Brands are offering less than 10% discounts.

# #### BRAND VS PRODUCT_NAME VS STAR_RATING
df.groupby(["BRAND","PRODUCT_NAME"]).describe()["STAR_RATING"].T

                              #Observations
# - It is observed that Apple with highest rating 4.7 for all of its products.
# - It is also Observed that from the Brand Asus is with greater than 4 star rating.
# - It is also observed that the Brands like Samsung ,Vivo,Oppo,Realme,Redmi,Poco,Infocus,Infinix are with 4 and above 4 rating.
# - The rest brands with 3.7 and 3.5 below ratings.

# ## Data Analysis
df.info()
df
### To get the descriptive statastics of Numeric columns
df.describe()

# ### Observations
# - We observe that there is huge difference b/w mean and median for the column ratings
# - We observe there is not much difference b/w mean and median for the columns Screen_Size, Warranty.
# - We observe the standard deiviation for the columns Ratings is very high.

#Univariate Analysis
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df.dtypes

# ### Memory
df['MEMORY'].value_counts()
sns.countplot(df['MEMORY'])

#Observations
# - We observe that Memory with 0 GB are higher in number followed by 64,128 GB ,32GB ,16GB and 256 GB
# - Because the in the above dataframe there were more feature mobiles(Feature mobiles donot have memory)
# - We also Observe that 512 GB are less than 5 mobiles

#### Star Rating
df['STAR_RATING'].value_counts()
plt.figure(figsize=(12,4))
sns.countplot(df['STAR_RATING'])

                           #Observations
# - It is observed that more than 90 mobiles are have 4.4 our 5 star ratings
# - It is observed that more than 80 mobiles are have 4.5 our 5 star ratings
# - It is observed that around 140 mobiles having 4.1 and 4.3 star ratings
# - There were other mobiles which were sharing ratings 3.8,3.7,3.9,4.0,3.6,3.5,etc ratings respectivels\y.
# - None of the mobiles have achived 5 star ratings from the above list
# - There were some mobiles which were not rated yet

# ### Battery capacity
df['BATTERY_C'].value_counts()
sns.boxplot(df['BATTERY_C'])

# #### Observations:
# - It is observed that mean is greater than mmedian and the skewed towards right side.

# ### Warranty
df['WARRANTY'].value_counts()
plt.figure(figsize=(3,6))
sns.countplot(df['WARRANTY'])

# #### Observations
# - It is observed that max. mobiles were offering only one year warranty.
# - And observed that less than 10% of mobiles are offering 2yr warranty.

# ### Reviews
sns.boxplot(df['REVIEWS'])
df['REVIEWS'].describe()

# #### Observations
# - It is observed that mean is very much higher than median.
# - The graph is skewed towards right side.
# - There were outliers present.

# ### Screen Size
sns.boxplot(df['SCREEN_SIZE'])

                     #Observations
# - we observe that mean is greater than median
# - The data is skewed towards right side.

# ### MRP
plt.figure(figsize=(18,4))
sns.boxplot(df['MRP'])

# #### Observations
# - We observe that the max MRP of the mobile exists till 150000.
# - The mean is very much higher than median
# - The data is skewed towards right side.
# - There were many outliers present for the column MRP.

# ### Offer Price

plt.figure(figsize=(18,4))
sns.boxplot(df['OFFER_PRICE'])

# #### Observations
# - We observe that the max offer-price of the mobile exists till 150000.
# - The mean is very much higher than median
# - The data is skewed towards right side.
# - There were many outliers present for the column Offerprice.
# - It is understood that offer-price is dependent on MRP .

df.groupby(by='BRAND').describe()

                  #Bivariate-Analysis
df.info()
plt.figure(figsize=(100,6))
sns.barplot(x='BRAND',y='WARRANTY',data=df)

                  #Observations:
plt.figure(figsize=(250,30))
sns.barplot(x='PRODUCT_NAME',y='MRP',data=df)
df.groupby(["BRAND"]).describe()["REVIEWS"].T
df.info()

# ##### BRAND VS MRP
plt.figure(figsize=(60,8))
sns.scatterplot(x="BRAND", y="MRP", data=df)

# ##### BRAND VS OFFER_PRCIE
plt.figure(figsize=(60,8))
sns.scatterplot(x="BRAND", y="OFFER_PRICE", data=df)

# ##### BRAND VS MEMORY
plt.figure(figsize=(60,8))
sns.scatterplot(x="BRAND", y="MEMORY", data=df)

# ##### BRAND VS MEMORY VS MRP
plt.figure(figsize=(60,8))
sns.scatterplot(x="BRAND", y="MEMORY", hue='MRP',data=df)

# #### BRAND VS MEMORY VS OFFER_PRICE
plt.figure(figsize=(60,8))
sns.scatterplot(x="BRAND", y="MEMORY", hue='OFFER_PRICE',data=df)

# #### BOX-PLOT for BRAND VS MRP
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="MRP", data =df);

# #### BOX-PLOT for BRAND VS OFFER_PRICE
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="OFFER_PRICE", data =df);

# #### BOX-PLOT for BRAND VS BATTERY_Capacity
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="BATTERY_C", data =df);

# #### BOX-PLOT for BRAND VS MEMORY
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="MEMORY", data =df);

# #### BOX-PLOT for BRAND VS WARRANTY
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="WARRANTY", data =df);

# #### BOX-PLOT for BRAND VS STAR_RATING
plt.figure(figsize=(70,10))
sns.boxplot(x="BRAND", y="STAR_RATING", data =df);
sns.pairplot(df)
df.corr()
sns.heatmap(df.corr(),annot=True)

                   #end of comparisions and visualizations







