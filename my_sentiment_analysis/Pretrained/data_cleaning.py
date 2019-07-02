# importing pandas package 
import pandas as pd 
  
# making data frame from csv file 
data = pd.read_csv('./data/BSLCMS-Comments.csv', usecols =['comments', 'comment_status'])
print(data.count())
# sorting by first name 
# data.sort_values("First Name", inplace = True) 
  
# dropping ALL duplicte values 
dropped_no = data.drop_duplicates(subset ="comments", 
                     keep = False, inplace = True) 
  
# displaying data 
print(data.count())



 # sir i want to know that how i shall get my money safely? when sebi send the refund form? if i attach my original document what will be the security with me till the receipt of money.please reply immediately 
 # sir i want to know that how i shall get my money safely? when sebi send the refund form? if i attach my original document what will be the security with me till the receipt of money.please reply immediately