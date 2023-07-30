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



def func(a):
    return a + a


def func3(x):
    return {0: 'setosa', 1: 'versicolor', 2: 'virginica', 5543: 'dominique'}[x]
    retlist = []
    for e in x:
        retlist.append({0: 'setosa', 1: 'versicolor',
                        2: 'virginica', 5543: 'dominique'}[e])
    return retlist


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
    try:
        oml.drop('tbl7')
        oml.drop('TBL6')
    except:
        pass
    cr = oml.cursor()
    v = list(map(lambda x: x.values(),artworkdata))
    
    
    df = pd.DataFrame({'numeric': [1, 1.4, -4, 3.145, 5, 2],
                   'string' : [None, None, 'a', 'a', 'a', 'b'],
                   'bytes' : [b'a', b'b', b'c', b'c', b'd', b'e']})
    oml_df3 = oml.create(df, table = "tbl7", 
                     dbtypes = {'numeric': 'BINARY_DOUBLE',
                                'bytes':'RAW(1)'})
    
    v = [("walter",1,"test0","test2"),"jennif",2,"test3","test4"]
    df = pd.DataFrame(
        {'artist_name':["walter","jennifer"],
         'chiartinstid':[1,2],
         'description':["test0","test3"],
         'title':["test2","test4"]}
    )
    try:
        proxy = oml.create(df,table ="TBL6",dbtypes={'artist_name':'VARCHAR2(4000)','chiartinstid':'NUMBER',
                                 'description':'VARCHAR2(4000)','title':'VARCHAR2(4000)'})
    except:
        oml.drop('tbl7')
        oml.drop('TBL6')
    digits = load_digits()
    pd_digits = pd.DataFrame(digits.data, columns=['IMG'+str(i) for i in range(digits['data'].shape[1])])
    pd_digits = pd.concat([pd_digits, pd.Series(digits.target,name = 'target')], axis = 1)

    try:
        oml.drop(table="DIGITS")
    except:
        pass
   
    DIGITS = oml.create(pd_digits, table = "DIGITS")
    cr.close()
    oml.disconnect()
    return
    
    
    testdata = {'artist_name': 'Philips Galle',
    'chiartinstid': 212977,
    'description': 'A work made of engraving in black on ivory laid paper.',
    'title': 'The Israelites Finding Achior Tied to a Tree, plate two from The'
            'Story of Judith and Holofernes'}
    
    
    v = oml.isconnected()
    v = oml.check_embed()
    
    try:
            oml.drop('tbl5')
    except:
        pass

    # Create a cursor object for the current OML4Py database 
    # connection to run queries and get information from the database.
    cr = oml.cursor()

    # Create a table from a list of tuples.
    lst = [(1, None, b'a'), (1.4, None, b'b'), (-4, 'a', b'c'), 
        (3.145, 'a', b'c'), (5, 'a', b'd'), (None, 'b', b'e')]
    oml_df5 = oml.create(lst, table = 'tbl5',
                        dbtypes = ['BINARY_DOUBLE','CHAR(1)','RAW(1)'])

    # Close the cursor
    cr.close()
    v = oml.ds.dir() 

    # Drop the tables.
    oml.drop('tbl5')
    oml.disconnect()
    
    
    
    
    # Notes from previous scratch
    #func2 = lambda a: a + a
    # print(func2(5))
    #ar = [0,1,2,5543]
    #val = map(func2,ar)
    # for v in val:
    #    pprint.pprint(v)
    #l = list(val)
    # The lambda function is just returning the value from the dictionary corresponding to the index passed
    # in the array entry. Note you can always fetch a value from a dictionary by passing the key value as an
    # index. Map function is used to fetch all values from the dictionary for each index value in the list that
    # is passed. The list function is used to build a list from all the values fetched from the dictionary
    #m = list(map(lambda x: {0: 'setosa', 1: 'versicolor', 2:'virginica',5543:'dominique'}[x],ar))
    #m = list(map(func3,ar))
    # for e in m:
    #    pprint.pp(e)


if __name__ == "__main__":
    main(sys.argv)
