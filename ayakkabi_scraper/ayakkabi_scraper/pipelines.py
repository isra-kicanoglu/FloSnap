import sqlite3
from scrapy.exceptions import DropItem

class DuplicatesPipeline:
    def __init__(self):
        self.con = sqlite3.connect('ayakkabilar.db')
        self.cur = self.con.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS scraped_items(
                product_id TEXT PRIMARY KEY
            )
        """)
        self.ids_seen = set()

    def open_spider(self, spider):
        self.cur.execute("SELECT product_id FROM scraped_items")
        for row in self.cur.fetchall():
            self.ids_seen.add(row[0])

    def process_item(self, item, spider):
        product_id = item.get('product_id')
        
        # Eğer product_id boşsa (None), bu item'ı atla
        if product_id is None:
            raise DropItem("Ürün ID'si bulunamadı, ürün atlanıyor.")
        
        if product_id in self.ids_seen:
            raise DropItem(f"Tekrar eden ürün bulundu: {product_id}")
        else:
            self.ids_seen.add(product_id)
            self.cur.execute("INSERT INTO scraped_items (product_id) VALUES (?)", (product_id,))
            self.con.commit()
            return item

    def close_spider(self, spider):
        self.con.close()