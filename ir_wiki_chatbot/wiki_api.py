# pip3 install wikipedia-api
# https://github.com/martin-majlis/Wikipedia-API/

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
        language='es',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

p_wiki = wiki_wiki.page("Elena Caffarena")
print(p_wiki.text)
