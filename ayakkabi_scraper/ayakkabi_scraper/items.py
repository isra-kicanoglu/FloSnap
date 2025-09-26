# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
import scrapy

class AyakkabiScraperItem(scrapy.Item):
    # Ayakkabının adı
    name = scrapy.Field()
    # Fiyatı
    price = scrapy.Field()
    url = scrapy.Field()
    # Markası
    brand = scrapy.Field()
    # Renk
    color = scrapy.Field()
    # Cinsiyet (erkek/kadın)
    gender = scrapy.Field()
    # Dış materyal
    outer_material = scrapy.Field()
    # İç materyal
    inner_material = scrapy.Field()
    # Bilek yüksekliği
    ankle_height = scrapy.Field()
    # Sezon bilgisi
    season = scrapy.Field()
    # Fotoğrafların URL'leri (4 açıdan)
    image_urls = scrapy.Field()

    ankle_height = scrapy.Field()
    # Diğer ek özellikler
    features = scrapy.Field()
   
    product_id = scrapy.Field()
