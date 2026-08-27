import pandas
import re

illegal_url_character = re.compile(r'[^A-Za-z0-9\-\._~:/\?#\[\]@!\$&\'\(\)\*\+,;=%]')

df = pandas.read_csv('temp_all_post_standart_time.csv')
print(df.columns)
all_urls = set()
for i in df['image_urls'].dropna().to_list():
    urls = i.split(', ')
    for url in urls:
        assert not illegal_url_character.findall(url), url
        all_urls.add(url)
for i in df['video_cover_url'].dropna().to_list():
    assert not illegal_url_character.findall(url), url
    all_urls.add(i)

with open('all_image_video_urls.txt', 'w', encoding='utf-8') as f:
    for url in all_urls:
        f.write(url + '\n')