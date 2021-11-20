# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:06:25 2021

@author: EricH
"""

                
import sys
import csv
from selenium import webdriver
import time

# default path to file to store data
path_to_file = "C:/Users/EricH/MachineLearning/try2/reviews.csv"

# default number of scraped pages
num_page = 10

# default tripadvisor website of hotel or things to do (attraction/monument) 
url = "https://www.tripadvisor.com/Hotel_Review-g60763-d1218720-Reviews-The_Standard_High_Line-New_York_City_New_York.html"
# if you pass the inputs in the command line
if (len(sys.argv) == 4):
    path_to_file = sys.argv[1]
    num_page = int(sys.argv[2])
    url = sys.argv[3]

# import the webdriver
driver = webdriver.Chrome()
driver.get(url)

# open the file to save the review
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)

# change the value inside the range to save more or less reviews
for i in range(0, num_page):

    # expand the review 
    time.sleep(2)
    driver.find_element_by_xpath(".//div[contains(@data-test-target, 'expand-review')]").click()
    container = driver.find_elements_by_xpath(".//div[@class='review-container']")
    dates = driver.find_elements_by_xpath(".//div[@class='_2fxQ4TOx']")

    for j in range(len(container)):
        review = container[j].find_element_by_xpath(".//q[@class='IRsGHoPm']").text.replace("  ")    
        csvWriter.writerow([review]) 
        
    # change the page            
    driver.find_element_by_xpath('.//a[@class="ui_button nav next primary "]').click()

driver.quit()

# from scrapy.spiders import Spider
# from scrapy.selector import Selector
# from scrapy.http import Request
# from scrapingtest import items
# from scrapingtest.items import ScrapingTestingItem



# class scrapingtestspider(Spider):
#     name = "scrapytesting"
#     allowed_domains = ["tripadvisor.in"]
#     base_uri = "tripadvisor.in"
#     start_urls = [
#         "http://www.tripadvisor.in/Hotel_Review-g297679-d300955-Reviews-Ooty_Fern_Hill_A_Sterling_Holidays_Resort-Ooty_Tamil_Nadu.html"]

#     output_json_dict = {}
#     def parse(self, response):

#         sel = Selector(response)
#         sites = sel.xpath('//a[contains(text(), "Next")]/@href').extract()
#         items = []
#         i=0
#         for sites in sites:
#             item = ScrapingTestingItem()
#             #item['reviews'] = sel.xpath('//p[@class="partial_entry"]/text()').extract()
#             item['subjects'] = sel.xpath('//span[@class="noQuotes"]/text()').extract()
#             item['stars'] = sel.xpath('//*[@class="rate sprite-rating_s rating_s"]/img/@alt').extract()
#             item['names'] = sel.xpath('//*[@class="username mo"]/span/text()').extract()
#             items.append(item)
#             i+=1
#             sites = sel.xpath('//a[contains(text(), "Next")]/@href').extract()

#             if(sites and len(sites) > 0):
#                 yield Request(url="tripadvisor.in" + sites[i], callback=self.parse)
#             else:
#                 yield 



import sys
import csv
from selenium import webdriver
import time

# default path to file to store data
path_to_file = "C:/Users/EricH/MachineLearning/try2/reviews3.csv"

# default number of scraped pages
num_page = 5

# default tripadvisor website of restaurant
url = "https://www.tripadvisor.com/Restaurant_Review-g60763-d802686-Reviews-Hard_Rock_Cafe-New_York_City_New_York.html"

# if you pass the inputs in the command line
if (len(sys.argv) == 4):
    path_to_file = sys.argv[1]
    num_page = int(sys.argv[2])
    url = sys.argv[3]

# Import the webdriver
driver = webdriver.Chrome()
driver.get(url)

# Open the file to save the review
csvFile = open(path_to_file, 'a', encoding="utf-8")
csvWriter = csv.writer(csvFile)

# change the value inside the range to save more or less reviews
for i in range(0, num_page):
    
    # expand the review 
    time.sleep(2)
    driver.find_element_by_xpath("//span[@class='taLnk ulBlueLinks']").click()

    container = driver.find_elements_by_xpath(".//div[@class='review-container']")

    for j in range(len(container)):

        title = container[j].find_element_by_xpath(".//span[@class='noQuotes']").text
        date = container[j].find_element_by_xpath(".//span[contains(@class, 'ratingDate')]").get_attribute("title")
        rating = container[j].find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute("class").split("_")[3]
        review = container[j].find_element_by_xpath(".//p[@class='partial_entry']").text.replace("\n", " ")

        csvWriter.writerow([date, rating, title, review]) 

    # change the page
    driver.find_element_by_xpath('.//a[@class="nav next ui_button primary"]').click()

driver.close()