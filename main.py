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
    oml.connect(user=omlusr,password=omlpass,dsn="aidb_medium",automl="aidb_medium_pool")
    pprint.pprint(oml.isconnected(check_automl=True))
    
    artworkdata = []
    # To avoid constantly hitting the API while testing, serialize first set of 
    # responses from API and use deserialized data. Delete objbuf.bin to fetch
    # fresh data.
    try:
        with open('objbuf.bin','rb') as f:
            artworkdata = pickle.load(f)
    except:
        pass
    
    if(len(artworkdata) == 0):
        agent_titles = []
        headers = {'user-agent': 'art-institute-browse/wgoulet@gmail.com'}
        url = "https://api.artic.edu/api/v1/agents?limit=100"
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        agents = r.json()
        agent_titles = list(map(lambda x: x['title'],agents['data']))
        loopmax = agents['pagination']['total']
        loopcount = 0
        buffersize = 100

        while(loopcount < loopmax):
            pagecount = agents['pagination']['total_pages']
            currentpage = agents['pagination']['current_page']
            while(pagecount != currentpage):
                url = agents['pagination']['next_url']
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                agents = r.json()
                agent_titles += list(map(lambda x: x['title'],agents['data']))
                currentpage = agents['pagination']['current_page']

            for artist_name in agent_titles:
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
                time.sleep(1)
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

            # Buffer first set of data returned so we can run this process again with small
            # amount of data stored for debugging without repeating whole process to fetch data again
            with open('objbuf.bin','wb') as f:
                pickle.dump(artworkdata,f)
            
            if((loopcount % buffersize == 0) or (loopmax - loopcount < buffersize)):
                pprint.pprint(f"buffering records{loopcount}")
                # Note for future self, if you use lowercase table names you have to wrap them
                # in "" when querying them with SQL. So use uppercase names for table names.
                # For Pandas data frames where we want to be able to provide column names
                # in the database, we need to change the shape of the data to pass it in 
                # as essentially a set of columns instead of rows. To illustrate,
                # we need to transform this shape:
                #    [
                #        {"title":"blah","artist_name":"blah","chiartinstid":0,"description":"blah"},
                #        {"title":"blahblah","artist_name":"blahblah","chiartinstid":1,"description":"blah"}
                #    ] 
                # into this shape:
                #     {"title":["blah","blahblah"],
                #      "chiartinstid":[0,1],
                #      "artist_name":["blah","blahblah"],
                #      "description":["blah","blah"]}
                #
                # The reason this shape is better is because when you pass data to OMLDB with
                # panda dataframes as a set of columns, you can specify the column name that the
                # data will be stored in as opposed to passing it lists of records and having the
                # database engine use autogenerated default column names. See
                # https://docs.oracle.com/en/database/oracle/machine-learning/oml4py/1/mlpug/get-started-oracle-machine-learning-python1.html#GUID-35E39AA2-3FF4-4F49-8556-E65967A52CA2
                
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

                oml_chiartdata = oml.create(df,table ="CHIARTDATA",dbtypes={'artist_name':'VARCHAR2(4000)','chiartinstid':'NUMBER',
                                            'description':'VARCHAR2(4000)','title':'VARCHAR2(4000)'},append=True)
                pprint.pprint(oml_chiartdata.head())
                chi_projected = oml_chiartdata[:,['description']]
                pprint.pprint(chi_projected.head(30))
                oml.disconnect()
                loopcount += 100
        

if __name__ == "__main__":
    main(sys.argv)
