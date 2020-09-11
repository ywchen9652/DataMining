# -*- coding: utf-8 -*-
"""
@author: hina
"""
print ()

import networkx
from operator import itemgetter
import matplotlib.pyplot
import pandas as pd

# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('./amazon-books.csv', index_col=0)

# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()

copurchaseGraph.edges

## Now let's assume a person is considering buying the following book;
## what else can we recommend to them based on copurchase behavior 
## we've seen from other users?
#print ("Looking for Recommendations for Customer Purchasing this Book:")
#print ("--------------------------------------------------------------")
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'

# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])

    

# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
# (1) 
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph.
purchasedAsinEgoGraph = networkx.Graph()
n=purchasedAsin
ego = networkx.ego_graph(copurchaseGraph,n,radius=1)
for f, t, e in ego.edges(data=True):
    purchasedAsinEgoGraph.add_edge(f,t,weight=e['weight'])
#
#
# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for f, t, e in purchasedAsinEgoGraph.edges(data=True):
    if e['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(f,t,weight=e['weight'])
        
# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = []
for i in  purchasedAsinEgoTrimGraph.neighbors(n):
    purchasedAsinNeighbors.append(i)

print(purchasedAsinNeighbors)

#
## Next, let's pick the Top Five book recommendations from among the 
## purchasedAsinNeighbors based on one or more of the following data of the 
## neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
## and ClusteringCoeff
## (4) 
##     Note that, given an asin, you can get at the metadata associated with  
##     it using amazonBooks (similar to lines 29-36 above).
##     Now, come up with a composite measure to make Top Five book 
##     recommendations based on one or more of the following metrics associated 
##     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
##     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
##     and include other measures if you like.
##     YOU MUST come up with a composite measure.
##     DO NOT simply make recommendations based on sorting!!!
##     Also, remember to transform the data appropriately using 
##     sklearn preprocessing so the composite measure isn't overwhelmed 
##     by measures which are on a higher scale.
#4-1.create a dataframe for purchasedAsinNeighbors and those correspoding metadata
Neighbors=[]
for i in purchasedAsinNeighbors:
    Neighbors.append(amazonBooks.loc[i,['Title','SalesRank','TotalReviews','AvgRating','DegreeCentrality','ClusteringCoeff']])
df_Neighbors=pd.DataFrame(data=Neighbors)
df_Neighbors

#4-2.standardize the values in ['TotalReviews','AvgRating','DegreeCentrality'] by MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler()
transform= mms.fit_transform(df_Neighbors[['TotalReviews','AvgRating','DegreeCentrality']])
df_transform=pd.DataFrame(transform,index=df_Neighbors.index, columns= ['mms_TotalReviews','mms_AvgRating','mms_DegreeCentrality'])
df_transform

#4-3.drop the original columns before standardization and conmbine the standardized columns
df_Neighbors_copy=df_Neighbors.copy()
df_Neighbors_drop=df_Neighbors_copy.drop(['TotalReviews','AvgRating','DegreeCentrality'],axis=1)
df_Neighbors_aftermms=pd.concat([df_Neighbors_drop,df_transform],axis=1)
df_Neighbors_aftermms

#4-4. divide 10000000 by the values in ['SalesRank'] to take the non-zero inverse values
df_Neighbors_aftermms['inverse_salesRank']= 10000000//df_Neighbors_aftermms['SalesRank']
df_Neighbors_aftermms['inverse_salesRank']
#and then use MinMaxScaler to standardize those values
from sklearn.preprocessing import MinMaxScaler
mms= MinMaxScaler()
mms_inverse_salesRank= mms.fit_transform(df_Neighbors_aftermms[['inverse_salesRank']])
df_Neighbors_aftermms['mms_inverse_salesRank']=pd.DataFrame(mms_inverse_salesRank,index=df_Neighbors_aftermms.index, columns=['inverse_salesRank'])
#drop the original column 
copy=df_Neighbors_aftermms.copy()
copy.drop(['SalesRank','inverse_salesRank'],axis=1,inplace=True)
copy

#4-5. add up the values in each row and present the value of the sum in a new column called "score"
df_Neighbors_aftermms=copy
df_Neighbors_aftermms['score']=df_Neighbors_aftermms['ClusteringCoeff']+df_Neighbors_aftermms['mms_inverse_salesRank']+df_Neighbors_aftermms['mms_TotalReviews']+df_Neighbors_aftermms['mms_AvgRating']+df_Neighbors_aftermms['mms_DegreeCentrality']
#and then sort the values by the score from high to low
df_Neighbors_aftermms.sort_values('score',inplace=True,ascending=False)
df_Neighbors_aftermms

#4-6. retrieve the top five row which have the highest five scores and present those values as a list
TOP5_ASIN=[]
for i in df_Neighbors_aftermms.head(n=5).index:
    TOP5_ASIN.append(i) 
    
TOP5_ASIN
#
# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)

for book_asin in TOP5_ASIN:
    print ("ASIN = ", book_asin) 
    print ("Title = ", amazonBooks.loc[book_asin,'Title'])
    print ("SalesRank = ", amazonBooks.loc[book_asin,'SalesRank'])
    print ("TotalReviews = ", amazonBooks.loc[book_asin,'TotalReviews'])
    print ("AvgRating = ", amazonBooks.loc[book_asin,'AvgRating'])
    print ("DegreeCentrality = ", amazonBooks.loc[book_asin,'DegreeCentrality'])
    print ("ClusteringCoeff = ", amazonBooks.loc[book_asin,'ClusteringCoeff'])
    print("-----------------")