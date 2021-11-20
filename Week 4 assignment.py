import pandas as pd

def trip_review_scraper(url_originale):
    import urllib 
    import progressbar
    import json
    import time
    from random import uniform
    from bs4 import BeautifulSoup
    import pandas as pd
    import math

#    import re
    def find_between( s, first, last ):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""
    index_pages_0 = url_originale
    
    page = urllib.request.urlopen(index_pages_0) #create the request to the server
    time.sleep(uniform(2,3)) #give the time to load the page
    soup = BeautifulSoup(page, 'html.parser') #extract the html text from the page
    
    #lets identify the number of total pages:
    v=soup.find_all('div',{"class": "pagination-details"})
    n_rec=[int(s) for s in (v[0].text).split() if s.isdigit()][2]
    num_pages=math.floor(n_rec/10) #each page has 10 reviews, so let's fix the number of iterations of the code
            
    #the link of each group of 10 reviews
    index_pages = []
    index_pages.append(index_pages_0)
    for i in range(1,int(num_pages)):
        pezzo=find_between(index_pages_0,'https','Reviews-')
        index_pages.append('https' + pezzo + 'Reviews-' +'or'+str(i*10) +'-' + index_pages_0.split('Reviews-',1)[1])      
    
    trip=[]
    trip_json=[]
    
    #now for each group of 10 reviews lets extract the link of the single review
    bar = progressbar.ProgressBar()
    for k in bar(range(0,len(index_pages))):
        page = urllib.request.urlopen(index_pages[k])
        time.sleep(uniform(2,4))
        soup = BeautifulSoup(page, 'html.parser') 
        
        #get the url in the pages and clean it keeping only the ones related to reviews
        link=[]
        for a in soup.find_all('a', href=True):
            link.append(a['href'])
        
        to_match=find_between( url_originale, 'Reviews-', '.html' )
        link_light=[f for f in link if to_match in f]
        link_2light=[f for f in link_light if 'ShowUserReviews' in f]
        
        titles=[]
        names=[]
        author_loc=[]
        date=[]
        contenuto=[]
        stars=[]
        
        for p in range(0,len(link_2light)):
            
            page = urllib.request.urlopen('https://www.tripadvisor.it/'+str(link_2light[p]))
            time.sleep(uniform(3,7))
            soup = BeautifulSoup(page, 'html.parser')
            
            #every review has a json format :) easy...
            script = soup.find('script', type='application/ld+json').text
            alfa= json.loads(str(script))
            trip_json.append(alfa)
              
            
            names.append(soup.find('span',{'class':'expand_inline scrname'}).text)  #author name
            author_loc.append(soup.find('span',{'class':'expand_inline userLocation'}).text) #author location
            
            date.append(soup.find('span',{"class": "ratingDate relativeDate"}).text) #review date

            contenuto.append(alfa['reviewBody']) #review body
            stars.append(int(alfa['reviewRating']['ratingValue'])) #rating
            titles.append(alfa['name']) #review title
            
        recensione=pd.DataFrame({'name':names,'author Location': author_loc, 'title':titles, 'date':date, 'rating':stars, 'text':contenuto})
        
        trip.append(recensione)
       # trip_json.append(alfa)
    return(trip,trip_json)

url_tripadvisor='https://www.tripadvisor.com/Hotel_Review-g187323-d638834-Reviews-Hotel_Berlin_Berlin-Berlin.html'
tripadvisor_review,review_json=trip_review_scraper(url_tripadvisor)
folder='files'
tripadvisor_review_df=pd.concat(tripadvisor_review)
tripadvisor_review_df.to_csv(folder+'/tripadvisor_reviw.csv')
with open(folder+'/tripadisor_reviews.json', 'w') as outfile:
    json.dump(review_json, outfile)



