# pip3 install wikipedia-api
# https://github.com/martin-majlis/Wikipedia-API/

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

p_wiki = wiki_wiki.page("cat")
print(p_wiki.text)
