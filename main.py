import sys
import pprint
import requests
import urllib
import time
import math
import pandas as pd
import oml
import os
import pickle
from sklearn.datasets import load_digits

def main(argv):
    omlusr = os.environ.get("OMLUSERNAME")
    omlpass = os.environ.get("OMLPASS")
    dbconn = oml.connect(user=omlusr,password=omlpass,dsn="aidb_medium")
    
    artworkdata = []
    try:
        with open('objbuf.bin','rb') as f:
            artworkdata = pickle.load(f)
    except:
        pass
    
    if(len(artworkdata) == 0):
        headers = {'user-agent': 'art-institute-browse/wgoulet@gmail.com'}

        url = "https://api.artic.edu/api/v1/agents?limit=100"
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        agents = r.json()

        for agent in agents['data']:
            artist_name = agent['title']
            print("Searching for artist {0} works".format(artist_name))
            

            # search only for individual artists
            searchqry = {
                "query": {
                    "bool": {
                        "must": {
                            "term": {
                                "agent_type_id": 7
                            }
                        },
                        "filter": {
                            "match": {
                                "title": {
                                    "query": "{0}".format(artist_name),
                                    "operator": "AND"
                                }
                            }
                        }
                    }
                }
            }

            codedqry = urllib.parse.urlencode(searchqry)
            url = "https://api.artic.edu/api/v1/agents/search?limit=100"
            r = requests.post(url, headers=headers, json=searchqry)
            r.raise_for_status()

            agents = r.json()
            if(len(agents['data']) == 0):
                continue
            artistid = agents['data'][0]['id']

            searchqry = {
                "query": {
                    "bool": {
                        "must": {
                            "term": {
                                "artist_id": artistid
                            }
                        }
                    }
                }
            }
            #time.sleep(0.25)
            url = "https://api.artic.edu/api/v1/artworks/search?limit=0"

            r = requests.post(url, headers=headers, json=searchqry)
            r.raise_for_status()
            art = r.json()
            resultsize = art['pagination']['total']
            pages = int(math.ceil(resultsize / 10))

            x = lambda a: "No description available" if(a == None) else a['alt_text']
            if(pages > 1):
                for i in range(1, min(pages,10)):
                    url = "https://api.artic.edu/api/v1/artworks/search?&limit=10&page={0}".format(
                        i)
                    r = requests.post(url, headers=headers, json=searchqry)
                    r.raise_for_status()
                    art = r.json()
                    for item in art['data']:
                        artworkdata.append({"title":item['title'],"artist_name":artist_name,
                                        "chiartinstid":item['id'],"description":x(item['thumbnail'])})
            else:
                url = "https://api.artic.edu/api/v1/artworks/search?&limit=10&page=1"
                r = requests.post(url, headers=headers, json=searchqry)
                r.raise_for_status()
                art = r.json()
                for item in art['data']:
                    artworkdata.append({"title":item['title'],"artist_name":artist_name,
                                        "chiartinstid":item['id'],"description":x(item['thumbnail'])})

        with open('objbuf.bin','wb') as f:
            pickle.dump(artworkdata,f)
            
    pprint.pprint(artworkdata)
    # Note for future self, if you use lowercase table names you have to wrap them
    # in "" when querying them with SQL. So use uppercase names for table names.
    try:
        oml.drop('CHIARTDATA')
    except:
        pass
    
    anamelist = list(map(lambda x: (x['artist_name']),artworkdata))
    idlist = list(map(lambda x: (x['chiartinstid']),artworkdata))
    desclist = list(map(lambda x: (x['description']),artworkdata))
    titlelist = list(map(lambda x: (x['title']),artworkdata))
    df = pd.DataFrame(
        {'artist_name':anamelist,
         'chiartinstid':idlist,
         'description':desclist,
         'title':titlelist}
    )

    proxy = oml.create(df,table ="CHIARTDATA",dbtypes={'artist_name':'VARCHAR2(4000)','chiartinstid':'NUMBER',
                                 'description':'VARCHAR2(4000)','title':'VARCHAR2(4000)'})
    oml.disconnect()

if __name__ == "__main__":
    main(sys.argv)
