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
import logging
import logging.handlers
import cx_Oracle

def main(argv):
    omlusr = os.environ.get("OMLUSERNAME")
    omlpass = os.environ.get("OMLPASS")
    oml.connect(user=omlusr,password=omlpass,dsn="aidb_medium",automl="aidb_medium_pool")
    oml_chiartdata = None
    pprint.pprint(oml.isconnected(check_automl=True))
    logger = logging.getLogger()
    handler = logging.handlers.SysLogHandler(address="/dev/log")
    logging.basicConfig()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    oml_chiartdata = oml.sync(schema="OMLUSER",table="CHIARTDATA")

    # Test selecting data
    x = oml_chiartdata.head(15)
    # Get list of artist names
    anames = x[:,'artist_name']
    # Get list of artistworks by artist name
    art = x[x['artist_name'] == "Peter Blume"]
    # Get list of artworks from artist as Panda dataframe
    adf = art.pull()
    # Get shape
    shape = adf.shape
    # Get entries from dataframe
    for i in range(0,shape[0]):
        record = adf.iloc[i]
        artist = record['artist_name']
        title = record['title']
        print(f"{artist} - {title}")
     


    
        

if __name__ == "__main__":
    main(sys.argv)
