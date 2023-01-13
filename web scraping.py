
                               #welcome to the  webscraping of mobiles
# importing the modules
import requests   #to get the request from website
from bs4 import BeautifulSoup #it is one of the webscraping tool
import re #that represents regular expressions
import pandas as pd #for data manipulation
cage=[]
soup=[]
Pr=[]
Mr=[]
Mpr=[]
R=[]
M=[]
baseurl ="/mobiles/pr?sid=tyy%2C4io&otracker=clp_metro_expandable_1_4.metroExpandable.METRO_EXPANDABLE_mobile-phones-store_25DMXHG2C5AT_wp3&fm=neo%2Fmerchandising&iid=M_fa5df413-888f-4a06-865a-4c4f6ca86973_4.25DMXHG2C5AT&ppt=clp&ppn=mobile-phones-store&ssid=hxiebd12w00000001586429271270&page="
num = int(input('Enter the number of pages for which you want links: '))
for i in range(num):
    url = 'https://www.flipkart.com/+'f'{baseurl}'+str(i+1)
    #print(url)
    cage.append(requests.get(url))
    soup=BeautifulSoup(cage[i].text)
    Pr.append(soup.find_all('div',attrs={'class':"_3wU53n"}))
    Mr.append(soup.find_all('div',attrs={'class':"_3ULzGw"}))
    Mpr.append(soup.find_all('div',attrs={'class':"_1-2Iqu row"}))
    R.append(soup.find_all('div',attrs={'class':"niH0FQ"}))
    M.append(soup.find_all('div',attrs={'class':"_6BWGkk"}))

                               #Observations:
# - We are gathering the information of each page and dividing the classes and storing the information in to lists.
# - Where each list of starting cell consisting of page1 information.
# - 2 cell of list containing of page 2 information and 3 cell containing of page 3 information and so one.    
# ### Product Brand,Name,Color
PBNCM=[]
for i in range(num):
    PBNCM=PBNCM+Pr[i]


PBNCM[0].text.strip()

PBNCM=list(map(lambda x: x.text.strip().replace('\n',' '),PBNCM))
print(len(PBNCM))
print(PBNCM)

import numpy as np
# ### feature 1,feature-2,feature3
B_brand=[]
P_product_name=[]
C_color=[]
for i in PBNCM:
    Brand=re.findall(r'(\w+).*',i)
    P_name=re.findall(r'\w*(\w*.*)',i)
    Col=re.findall(r'\((\w*.*),',i)
    if len(Brand)==0:
        B_brand.append(np.nan)
    else:
        B_brand.append(Brand[0])
    if len(P_name)==0:
        P_product_name.append(np.nan)
    else:
        P_product_name.append(P_name[0])
    if len(Col)==0:
        C_color.append(np.nan)
    else:
        C_color.append(Col[0])
print(len(P_product_name))
print(P_product_name)
print(len(C_color))
print(C_color)
print(len(B_brand))
print(B_brand)
                     # Observations
# - We are gathering the information from the Product name class.
# - Here we are getting 3 features
#     - 1.Brand
#     - 2.Product model name
#     - 3.Product Color

# ### Rom,Ram,Screen_Size
Rom=[]
for i in range(num):
    Rom=Rom+Mpr[i]

Rom=list(map(lambda x: x.text.strip().replace('\n',' '),Rom))
print(len(Rom))
print(Rom)
# ### features-4,5,6
Rom_m=[]
Ram_m=[]
S_size=[]
for i in Rom:
    Ro=re.findall(r'(\d+) GB ROM',i)
    Ro=["0" if x =='' else x for x in Ro]
    Ra=re.findall(r'(\d+) GB RAM',i)
    Ra=["0" if x =='' else x for x in Ra]
    SS=re.findall(r'(\d+.\d+) inch',i)
    SS=["0" if x =='' else x for x in SS]
    if len(Ro)==0:
        Rom_m.append(np.nan)
    else:
        Rom_m.append(Ro[0])
    if len(Ra)==0:
        Ram_m.append(np.nan)
    else:
        Ram_m.append(Ra[0])
    if len(SS)==0:
        S_size.append(np.nan)
    else:
        S_size.append(SS[0])
print(len(Rom_m))
print(Rom_m)
print(len(Ram_m))
print(Ram_m)
print(len(S_size))
print(S_size)


                        #Observations
# - We are gathering some features from the specifications class are listed below
#     - 1.Ram
#     - 2.Rom
#     - 3.Screen size in inches

# ### Rating,no. of Ratings,no of Reviews
RNRR=[]
for i in range(num):
    RNRR=RNRR+Mpr[i]
# ### Feature 7
# finding and validating Rating for Each mobile
RNRRM = []
for i in RNRR:
    Nri = i.find_all('div',class_='hGSR34')
    if len(Nri) == 0:
        RNRRM.append(np.nan)
    else:
        RNRRM.append(Nri[0].text)
        
print(len(RNRRM))
print(RNRRM)

RR=[]
for i in range(num):
    RR=RR+Mpr[i]

# finding and validating discount if it's present for a mobile
import numpy as np
sc = []
for i in RR:
    c = i.find_all('span',class_='_38sUEc')
    if len(c) == 0:
        sc.append('')
    else:
        c = c[0].text
        sc.append(c)
        
print(len(sc))
print(sc)

# ### Feature 8,9
### Rat,Rew
Rat=[]
Rew=[]
for i in sc:
    Ro=re.findall(r'(\d*[,]\d*[,]\d*|\d*[,]\d*) Ratings',i)
    Re=re.findall(r'(\d*[,]\d*[,]\d*|\d*[,]\d*) Reviews',i)
    if len(Ro)==0:
        Rat.append(np.nan)
    else:
        Rat.append(Ro[0])
    if len(Re)==0:
        Rew.append(np.nan)
    else:
        Rew.append(Re[0])
        
print(len(Rat))
print(Rat)
print(len(Rew))
print(Rew)


                         #Observations
# - Gathering information from the ratings class.
# - Here we can get 3 features.
#     1.Star rating
#     2.No. of ratings
#     3.No. of reviews
# - For some products there were no ratings there we substituting nan vales where the star-rating,no.of ratings and no.of revies are missing.

# ### Battery,Warranty
RBSW=[]
for i in range(num):
    RBSW=RBSW+Mr[i]
    
RBSW[0]
RBSW[23].text.strip()
RBSW=list(map(lambda x: x.text.strip().replace('\n',' '),RBSW))
print(len(RBSW))
print(RBSW)
# ### Feature 10,11
### Battery,Warranty
Bat=[]
Wat=[]
for i in RBSW:
    bo=re.findall(r'(\d*) mAh',i)
    wr=re.findall(r'(\d|\d.\d) Year',i)
    if len(bo)==0:
        Bat.append('0')
    else:
        Bat.append(bo[0])
    if len(wr)==0:
        Wat.append(np.nan)
    else:
        Wat.append(wr[0])
        
print(len(Bat))
print(Bat)
print(len(Wat))
print(Wat)


                           #Observations
# - From specifications clas we are extracting Battery capacity and Warranty .
# - Some products does not containing no battery capacity, there we cannot assume the capacity ,so we are substituting nan values
# - For some produts there was no warranty that we are substituting nan values
# ### Discount,Offer price,Mrp
Di=[]
for i in range(num):
    Di=Di+M[i]

# ### Feature-12
# finding and validating discount if it's present for a mobile
Disco = []
for i in Di:
    di = i.find_all('div',class_='VGWI6T')
    if len(di) == 0:
        Disco.append('0 % off')
    else:
        Disco.append(di[0].text)
print(len(Disco))
print(Disco)

# ### Replacing 0% where there is no discount
# 
# - By replacing 0% where there is no discount offered we can know how many products were with out discounts.
# - There will be no choice of truncating of rows or columns for the discount
# finding and validating Offer_Price if it's present for a mobile
O_price = []
for i in Di:
    P_p = i.find_all('div',class_='_1uv9Cb')
    if len(P_p) == 0:
        O_price.append(np.nan)
    else:
        O_price.append(P_p[0].text)
        
print(len(O_price))
print(O_price)
# ### Feature 13
OP1=[]
for i in O_price:
    op_1=re.findall(r'₹d*.*₹|₹\d*.*',i)
    if len(op_1)==0:
        OP1.append(np.nan)
    else:
        OP1.append(op_1[0])
OP1 = [re.sub('[₹,]', '', item) for item in OP1]
print(len(OP1))
print(OP1)
# ### Feature-14
# Mrp
MP1=[]
j=0
for i in O_price:
    mp_1=re.findall(r'(\d+,\d{3}|\d+,\d+,\d{3})\d+%',i)
    if len(mp_1)==0:
        MP1.append(OP1[j])
        j=j+1
    else:
        MP1.append(mp_1[0])
        j=j+1
        
MP1 = [re.sub('[,]', '', item) for item in MP1]
print(len(MP1))
print(MP1)

                      #Observations from website 
# - It is observed that offer price of some products is Missing.
# - In this case we are considering Mrp as offerprice.
# - For these products there will be no discount offered which we replaced 0% discount in the discount class.

# Structuring all the columns in to a DataFrame
Fmobiles=pd.DataFrame()
Fmobiles['BRAND']=B_brand
Fmobiles['PRODUCT_NAME']=P_product_name
Fmobiles['COLOR']=C_color
Fmobiles['MEMORY']=Rom_m
Fmobiles['STAR_RATING']=RNRRM
Fmobiles['RATINGS']=Rat
Fmobiles['REVIEWS']=Rew
Fmobiles['RAM']=Ram_m
Fmobiles['BATTERY_C']=Bat
Fmobiles['SCREEN_SIZE']=S_size
Fmobiles['WARRANTY']=Wat
Fmobiles['OFFER_PRICE']=OP1
Fmobiles['DISCOUNT']=Disco
Fmobiles['MRP']=MP1

Fmobiles.head()

Fmobiles.info()

Fmobiles.to_csv('Fmobiles.csv')

                   #End of webscraping (stored data in the form of csv)






