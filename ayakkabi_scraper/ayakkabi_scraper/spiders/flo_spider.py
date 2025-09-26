import scrapy
from scrapy_splash import SplashRequest
from scrapy.exceptions import CloseSpider
import re
from ..items import AyakkabiScraperItem

class FloSpider(scrapy.Spider):
    name = "flo"

    # Splash için gerekli Lua script'leri  #sonsuz kaydırma sağlar
    lua_script_main = """
    function main(splash, args)
        splash:set_viewport_full()
        assert(splash:go(args.url))
        splash:wait(5)
        local last_height = splash:evaljs('document.body.scrollHeight')
        while true do
            splash:runjs('window.scrollTo(0, document.body.scrollHeight)')
            splash:wait(3)
            local new_height = splash:evaljs('document.body.scrollHeight')
            if new_height == last_height then
                break
            end
            last_height = new_height
        end
        return splash:html()
    end
    """
    #her kaydırmada verilerin detayını çeker
    lua_script_details = """ 
    function main(splash, args)
        splash:set_viewport_full()
        assert(splash:go(args.url))
        splash:wait(2.5)
        local images = splash:evaljs("window.productDetailModel && window.productDetailModel.images;")
        
        if not images then
            images = {}
        end

        return {
            html = splash:html(),
            images = images
        }
    end
    """
    
    # Çoklu sayfa çekmek için start_requests metodunu güncelliyoruz
    def start_requests(self):
        # 10 sayfa gezmek için range(1, 11) kullanabilirsin
        # 100 ürün için 1-5 aralığı genelde yeterli olur (5 x 24 = 120 ürün)
        for page_number in range(1, 10):
            url = f"https://www.flo.com.tr/ayakkabi?cinsiyet=erkek,kadin&sort=last_order_count:desc&marka=nike,puma,adidas&page=2&page={page_number}"
            yield SplashRequest(
                url=url,
                callback=self.parse_product_links,
                endpoint='execute',
                args={'lua_source': self.lua_script_main, 'timeout': 90}
            )  #90 sn içinde yanıt gelmezse durdur

    def parse_product_links(self, response): #her bi ürünü detay sayfasını yönlendiriyo(yield ile)
        product_links = response.css('a[data-test="open-to-product-detail-from-list"]::attr(href)').getall()
        
        self.log(f"{len(product_links)} adet ürün linki bulundu!")
        
        for link in product_links:
            yield SplashRequest(
                url=response.urljoin(link),
                callback=self.parse_product_details,
                endpoint='execute',
                args={'lua_source': self.lua_script_details, 'timeout': 90}
            )

    def parse_product_details(self, response):
        # NOT: Pipeline'lar artık ürün sayısını kontrol ettiği için bu kodlara gerek yok.
        # Bu satırları siliyoruz:
        # self.item_count += 1
        # if self.item_count >= self.max_items:
        #    self.log(f"Hedeflenen {self.max_items} ürüne ulaşıldı. Spider durduruluyor.")
        #    raise CloseSpider("Hedeflenen ürün sayısına ulaşıldı.")

        self.log(f"Ürün çekiliyor: {response.url}")
        #sayfadaki o elementin en belirgin özelliği kullandım 
        full_name = response.css('h1.product-detail__name span.js-product-name::text').get()
        brand = response.css('a.product-detail__brand::text').get()
        price = response.css('input.js-bunsar-price::attr(value)').get()

        url_parts = response.url.split('-')
        product_id = url_parts[-1] if url_parts and url_parts[-1].isdigit() else None
        
        details = {}
        detail_items = response.css('div.detail-properties__item')
        
        for item in detail_items:
            key = item.css('span.detail-properties__item-label::text').get()
            value = item.css('span.detail-properties__item-value::text').get()
            if key and value:
                clean_key = key.strip().lower().replace(' ', '_').replace('.', '').replace('i̇', 'i')
                details[clean_key] = value.strip()
        
        product_info_text = response.css('div.product-detail__info-content p::text').get()
        if product_info_text:
            text_lower = product_info_text.lower()
            inner_material_match = re.search(r'i̇ç materyal:?\s*([a-zğüşöç\s]+)', text_lower)
            if inner_material_match:  #ürün açıklama kısmında buldum etiket bu
                details['iç_materyal'] = inner_material_match.group(1).strip().title()

        all_image_urls = []
        
        raw_image_data = response.data.get('images', [])
        all_image_urls.extend([response.urljoin(item.get('href')) for item in raw_image_data if isinstance(item, dict) and item.get('href')])
        
        all_image_urls.extend(response.css('a.detail__images-item::attr(href)').getall())
        all_image_urls.extend(response.css('img.swiper-lazy::attr(src)').getall())

        valid_images = [url for url in all_image_urls if 'floimages.mncdn.com' in url]
        unique_images = list(dict.fromkeys(valid_images))
        
        item = AyakkabiScraperItem(
            product_id=product_id,
            brand=brand.strip() if brand else None,
            name=full_name.strip() if full_name else None,
            price=price,
            url=response.url,
            color=details.get('renk'),
            gender=details.get('cinsiyet'),
            outer_material=details.get('dış_materyal'),
            inner_material=details.get('iç_materyal'),
            ankle_height=details.get('bilek_yüksekliği'),
            image_urls=unique_images[:4]
        )
        yield item