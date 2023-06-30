import quandl

api_key = 'your_api_key'
quandl.ApiConfig.api_key = api_key

data = quandl.get('WIKI/FB', start_date='2018-01-01', end_date='2018-12-31')

print(data)
