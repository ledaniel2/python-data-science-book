import scrapy

class YCombinatorSpider(scrapy.Spider):
    name = 'ycombinator_news'
    allowed_domains = ['news.ycombinator.com']
    start_urls = ['https://news.ycombinator.com/']

    def parse(self, response):
        for article in response.css('span.titleline'):
            yield {
                'title': article.css('a::text').get(),
                'url': article.css('a::attr(href)').get(),
            }

        next_page = response.css('a.morelink::attr(href)').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
