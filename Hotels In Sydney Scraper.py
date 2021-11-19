# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:06:25 2021

@author: EricH
"""



# const Apify = require('apify');

# const { utils: { log } } = Apify;
# const general = require('./general');

# const { getReviews, getReviewTags, randomDelay } = general;
# const { getPlacePrices } = require('./api');
# const { incrementSavedItems, checkMaxItemsLimit } = require('./data-limits');

# /**
#  *
#  * @param {{
#  *   placeInfo: unknown,
#  *   client: general.Client,
#  *   dataset?: Apify.Dataset,
#  *   session: Apify.Session,
#  * }} params
#  */
# async function processHotel({ placeInfo, client, dataset, session }) {
#     const { location_id: id } = placeInfo;
#     let reviews = [];
#     const placePrices = {};

#     try {
#         // placePrices = await getPlacePrices({ placeId: id, delay: randomDelay, session });
#     } catch (e) {
#         log.warning('Hotels: Could not get place prices', { errorMessage: e });
#     }

#     if (global.INCLUDE_REVIEWS) {
#         try {
#             reviews = await getReviews({ placeId: id, client, session });
#         } catch (e) {
#             log.exception(e, 'Could not get reviews');
#             throw e;
#         }
#     }

#     if (!placeInfo) {
#         return;
#     }
#     const prices = placePrices?.offers?.map((offer) => ({
#         provider: offer.provider_display_name,
#         price: offer.display_price_int ? offer.display_price_int : 'NOT_PROVIDED',
#         isBookable: offer.is_bookable,
#         link: offer.link,
#     })) ?? [];

#     const place = {
#         id: placeInfo.location_id,
#         type: 'HOTEL',
#         name: placeInfo.name,
#         awards: placeInfo.awards?.map((award) => ({ year: award.year, name: award.display_name })) ?? [],
#         rankingPosition: placeInfo.ranking_position,
#         priceLevel: placeInfo.price_level,
#         priceRange: placeInfo.price,
#         category: placeInfo.ranking_category,
#         rating: placeInfo.rating,
#         hotelClass: placeInfo.hotel_class,
#         hotelClassAttribution: placeInfo.hotel_class_attribution,
#         phone: placeInfo.phone,
#         address: placeInfo.address,
#         email: placeInfo.email,
#         amenities: placeInfo.amenities?.map((amenity) => amenity.name) ?? [],
#         prices,
#         latitude: placeInfo.latitude,
#         longitude: placeInfo.longitude,
#         webUrl: placeInfo.web_url,
#         website: placeInfo.website,
#         rankingString: placeInfo.ranking,
#         rankingDenominator: placeInfo.ranking_denominator,
#         numberOfReviews: placeInfo.num_reviews,
#         reviewsCount: reviews.length,
#         reviews,
#     };
#     if (global.INCLUDE_REVIEW_TAGS) {
#         const tags = await getReviewTags({ locationId: id, session });
#         place.reviewTags = tags;
#     }
#     log.debug(`Data for hotel: ${place.name}`);
#     if (dataset) {
#         if (!checkMaxItemsLimit()) {
#             await dataset.pushData(place);
#             incrementSavedItems();
#         }
#     } else {
#         await Apify.setValue('OUTPUT', JSON.stringify(place), { contentType: 'application/json' });
#     }
# }

# module.exports = {
#     processHotel,
# };   
# import re
# import time

# from scrapy.spider import BaseSpider
# from scrapy.selector import Selector
# from scrapy.http import Request

# from tripadvisorbot.items import *
# from tripadvisorbot.spiders.crawlerhelper import *


# # Constants.
# # Max reviews pages to crawl.
# # Reviews collected are around: 5 * MAX_REVIEWS_PAGES
# MAX_REVIEWS_PAGES = 500


# class TripAdvisorRestaurantBaseSpider(BaseSpider):
# 	name = "tripadvisor-restaurant"

# 	allowed_domains = ["tripadvisor.com"]
# 	base_uri = "http://www.tripadvisor.com"
# 	start_urls = [
# 		base_uri + "/RestaurantSearch?geo=60763&q=New+York+City%2C+New+York&cat=&pid="
# 	]


# 	# Entry point for BaseSpider.
# 	# Page type: /RestaurantSearch
# 	def parse(self, response):
# 		tripadvisor_items = []

# 		sel = Selector(response)
# 		snode_restaurants = sel.xpath('//div[@id="EATERY_SEARCH_RESULTS"]/div[starts-with(@class, "listing")]')
# 		
# 		# Build item index.
# 		for snode_restaurant in snode_restaurants:

# 			tripadvisor_item = TripAdvisorItem()

# 			tripadvisor_item['url'] = self.base_uri + clean_parsed_string(get_parsed_string(snode_restaurant, 'div[@class="quality easyClear"]/span/a[@class="property_title "]/@href'))
# 			tripadvisor_item['name'] = clean_parsed_string(get_parsed_string(snode_restaurant, 'div[@class="quality easyClear"]/span/a[@class="property_title "]/text()'))
# 			
# 			# Cleaning string and taking only the first part before whitespace.
# 			snode_restaurant_item_avg_stars = clean_parsed_string(get_parsed_string(snode_restaurant, 'div[@class="wrap"]/div[@class="entry wrap"]/div[@class="description"]/div[@class="wrap"]/div[@class="rs rating"]/span[starts-with(@class, "rate")]/img[@class="sprite-ratings"]/@alt'))
# 			tripadvisor_item['avg_stars'] = re.match(r'(\S+)', snode_restaurant_item_avg_stars).group()

# 			# Popolate reviews and address for current item.
# 			yield Request(url=tripadvisor_item['url'], meta={'tripadvisor_item': tripadvisor_item}, callback=self.parse_search_page)

# 			tripadvisor_items.append(tripadvisor_item)
# 		

# 	# Popolate reviews and address in item index for a single item.
# 	# Page type: /Restaurant_Review
# 	def parse_search_page(self, response):
# 		tripadvisor_item = response.meta['tripadvisor_item']
# 		sel = Selector(response)


# 		# TripAdvisor address for item.
# 		snode_address = sel.xpath('//div[@class="wrap infoBox"]')
# 		tripadvisor_address_item = TripAdvisorAddressItem()

# 		tripadvisor_address_item['street'] = clean_parsed_string(get_parsed_string(snode_address, 'address/span/span[@class="format_address"]/span[@class="street-address"]/text()'))

# 		snode_address_postal_code = clean_parsed_string(get_parsed_string(snode_address, 'address/span/span[@class="format_address"]/span[@class="locality"]/span[@property="v:postal-code"]/text()'))
# 		if snode_address_postal_code:
# 			tripadvisor_address_item['postal_code'] = snode_address_postal_code

# 		snode_address_locality = clean_parsed_string(get_parsed_string(snode_address, 'address/span/span[@class="format_address"]/span[@class="locality"]/span[@property="v:locality"]/text()'))
# 		if snode_address_locality:
# 			tripadvisor_address_item['locality'] = snode_address_locality

# 		tripadvisor_address_item['country'] = clean_parsed_string(get_parsed_string(snode_address, 'address/span/span[@class="format_address"]/span[@class="locality"]/span[@property="v:region"]/text()'))
# 		
# 		tripadvisor_item['address'] = tripadvisor_address_item


# 		# TripAdvisor photos for item.
# 		tripadvisor_item['photos'] = []
# 		snode_main_photo = sel.xpath('//div[@class="photoGrid photoBx"]')

# 		snode_main_photo_url = clean_parsed_string(get_parsed_string(snode_main_photo, 'div[starts-with(@class, "photo ")]/a/@href'))
# 		if snode_main_photo_url:
# 			yield Request(url=self.base_uri + snode_main_photo_url, meta={'tripadvisor_item': tripadvisor_item}, callback=self.parse_fetch_photo)


# 		tripadvisor_item['reviews'] = []

# 		# The default page contains the reviews but the reviews are shrink and need to click 'More' to view the complete content.
# 		# An alternate way is to click one of the reviews in the page to open the expanded reviews display page.
# 		# We're using this last solution to avoid AJAX here.
# 		expanded_review_url = clean_parsed_string(get_parsed_string(sel, '//div[contains(@class, "basic_review")]//a/@href'))
# 		if expanded_review_url:
# 			yield Request(url=self.base_uri + expanded_review_url, meta={'tripadvisor_item': tripadvisor_item, 'counter_page_review' : 0}, callback=self.parse_fetch_review)


# 	# If the page is not a basic review page, we can proceed with parsing the expanded reviews.
# 	# Page type: /ShowUserReviews
# 	def parse_fetch_review(self, response):
# 		tripadvisor_item = response.meta['tripadvisor_item']
# 		sel = Selector(response)

# 		counter_page_review = response.meta['counter_page_review']

# 		# Limit max reviews pages to crawl.
# 		if counter_page_review < MAX_REVIEWS_PAGES:
# 			counter_page_review = counter_page_review + 1

# 			# TripAdvisor reviews for item.
# 			snode_reviews = sel.xpath('//div[@id="REVIEWS"]/div/div[contains(@class, "review")]/div[@class="col2of2"]/div[@class="innerBubble"]')

# 			# Reviews for item.
# 			for snode_review in snode_reviews:
# 				tripadvisor_review_item = TripAdvisorReviewItem()
# 				
# 				tripadvisor_review_item['title'] = clean_parsed_string(get_parsed_string(snode_review, 'div[@class="quote"]/text()'))

# 				# Review item description is a list of strings.
# 				# Strings in list are generated parsing user intentional newline. DOM: <br>
# 				tripadvisor_review_item['description'] = get_parsed_string_multiple(snode_review, 'div[@class="entry"]/p/text()')
# 				# Cleaning string and taking only the first part before whitespace.
# 				snode_review_item_stars = clean_parsed_string(get_parsed_string(snode_review, 'div[@class="rating reviewItemInline"]/span[starts-with(@class, "rate")]/img/@alt'))
# 				tripadvisor_review_item['stars'] = re.match(r'(\S+)', snode_review_item_stars).group()
# 				
# 				snode_review_item_date = clean_parsed_string(get_parsed_string(snode_review, 'div[@class="rating reviewItemInline"]/span[@class="ratingDate"]/text()'))
# 				snode_review_item_date = re.sub(r'Reviewed ', '', snode_review_item_date, flags=re.IGNORECASE)
# 				snode_review_item_date = time.strptime(snode_review_item_date, '%B %d, %Y') if snode_review_item_date else None
# 				tripadvisor_review_item['date'] = time.strftime('%Y-%m-%d', snode_review_item_date) if snode_review_item_date else None

# 				tripadvisor_item['reviews'].append(tripadvisor_review_item)


# 			# Find the next page link if available and go on.
# 			next_page_url = clean_parsed_string(get_parsed_string(sel, '//a[starts-with(@class, "guiArw sprite-pageNext ")]/@href'))
# 			if next_page_url and len(next_page_url) > 0:
# 				yield Request(url=self.base_uri + next_page_url, meta={'tripadvisor_item': tripadvisor_item, 'counter_page_review' : counter_page_review}, callback=self.parse_fetch_review)
# 			else:
# 				yield tripadvisor_item

# 		# Limitatore numero di pagine di review da passare. Totale review circa 5*N.
# 		else:
# 			yield tripadvisor_item


# 	# Popolate photos for a single item.
# 	# Page type: /LocationPhotoDirectLink
# 	def parse_fetch_photo(self, response):
# 		tripadvisor_item = response.meta['tripadvisor_item']
# 		sel = Selector(response)

# 		# TripAdvisor photos for item.
# 		snode_photos = sel.xpath('//img[@class="taLnk big_photo"]')

# 		# Photos for item.
# 		for snode_photo in snode_photos:
# 			tripadvisor_photo_item = TripAdvisorPhotoItem()

# 			snode_photo_url = clean_parsed_string(get_parsed_string(snode_photo, '@src'))
# 			if snode_photo_url:
# 				tripadvisor_photo_item['url'] = snode_photo_url
# 				tripadvisor_item['photos'].append(tripadvisor_photo_item)
                
                
                
                
                
                
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

    container = driver.find_elements_by_xpath("//div[@data-review]")
    dates = driver.find_elements_by_xpath(".//div[@class='_2fxQ4TOx']")

    for j in range(len(container)):
        review = container[j].find_element_by_xpath(".//q[@class='IRsGHoPm']").text.replace("\n", "  ")    
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
path_to_file = "C:/Users/EricH/MachineLearning/try2/reviews.csv"

# default number of scraped pages
num_page = 10

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