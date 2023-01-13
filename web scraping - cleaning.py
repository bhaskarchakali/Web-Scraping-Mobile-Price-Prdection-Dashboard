
#Importing necessary libraries
import numpy as np         # for numerical calculations
import pandas as pd        # for data manipulation
df=pd.read_csv('C:/Users/Home/Desktop/Interview project/Fmobiles.csv')
df.head()                  #by default it gives top 5 rows informnation
# We observe that all the columns of the object datatype. We have to change some of them in to numeric datatypes.
# - Columns to be changed: 
#  `MEMORY`,`STAR_RATING`,`RATINGS`,`REVIEWS`,`RAM`,`BATTERY_C`,`SCREEN_SIZE`,`OFFER_PRICE`,`DISCOUNT`,`MRP`

df['OFFER_PRICE']=df['OFFER_PRICE'].astype(int)
df['MRP']=df['MRP'].astype(int)
      #Observation
# Offerprice and Mrp can be converted to int type where there are no null values and there are no float values

df['REVIEWS']=df['REVIEWS'].str.replace(',','').astype(float)
df['MEMORY']=df['MEMORY'].astype(float)
df['RATINGS']=df['RATINGS'].str.replace(',','').astype(float)
df['RAM']=df['RAM'].astype(float)
df['BATTERY_C']=df['BATTERY_C'].astype(float)
df['STAR_RATING']=df['STAR_RATING'].astype(float)
df['SCREEN_SIZE']=df['SCREEN_SIZE'].astype(float)
df['WARRANTY']=df['WARRANTY'].astype(float)

        # Observations:
# `Reviews`,`Ratings`,`Ram`,`Battery_c`,`Star_Rating`,`Screen size`,`Warranty`, are containing nan values.Where nan is float valriable so need to convert them to float values

df.info()
df.drop(axis=1,columns=['Unnamed: 0'],inplace=True)
df.head()
                 #Observations:
# - While import the CSV file we found that index column is added to dataframe.
# - We drop the unnamed column us drop() method.
df.info()
# Finding the null values:
df.isnull()
#To know the total values which are missing in the dataframe,column wise
df.isnull().sum()
#Checking for the total no of missing values 
((df.isnull().sum())/len(df))*100
# ### Product name:
df.dropna(inplace=True,subset=['PRODUCT_NAME'])
df.info()
# #### Observations
# Since Product name  is unique,replacing it with some value doesn't make right so we drop it

df['WARRANTY'].value_counts().index[0]
# #### Warranty
df['WARRANTY'].fillna(df['WARRANTY'].value_counts().index[0],inplace=True)

                   #Observations:
# - Let's impute Warranty with the most frequently occuring value.
# - Because warranty is containg less no. of missing values and max. mobiles are have min warranty so we are filling the data with the most frequently occuring value.

df.describe()
# ### STAR_RATING
df['STAR_RATING']=df['STAR_RATING'].fillna(0)

                     #Observations:
# - Since some mobiles are not having star rating that means those were not rated yet.
# - We can assume zero rating rather than deleting the column
df.info()
# ### Memory
df['MEMORY']=df['MEMORY'].fillna(0)

                                   #Observations:
# - We observe that some mobiles are not having Memory because for feature mobiles Memory will not present.
# - We can assume zero as memory rather than deleting the column
# ### RAM
df['RAM']=df['RAM'].fillna(0)

                               #Observations:
# - We observe that some mobiles are not having RAM because for feature mobiles RAM will not present as like Memory.
# - We can assume zero for RAM rather than deleting the column

# RATINGS
df['RATINGS']=df['RATINGS'].fillna(0)
                             #Observations:
# - Since some mobiles are not having ratings that means those were not rated yet.
# - We can assume zero rating rather than deleting the column

# ### REVIEWS
df['REVIEWS']=df['REVIEWS'].fillna(0)
                         #Observations:
# - Since some mobiles are not having Reviews that means those were not reviewed yet.
# - We can assume zero reviews rather than deleting the column

df.info()
df['SCREEN_SIZE'].describe()
df['SCREEN_SIZE'].fillna(df.describe().loc['mean','SCREEN_SIZE'],inplace=True)
df.info()
df['COLOR']=df['COLOR'].fillna('No Color')
df.info()
df['MEMORY']=df['MEMORY'].astype(int)
df['REVIEWS']=df['REVIEWS'].astype(int)
df['RATINGS']=df['RATINGS'].astype(int)
df.to_csv('Fmobiles_clean.csv')

                           #end of scraped data cleaning







