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
from oml import cursor
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
    # split data
    set = oml_chiartdata.tail(150).split(ratio=(.25,.25),use_hash=False) 
    set

    # Create training data and test data.
    dat = oml.push(pd.DataFrame( 
    {'COMMENTS':['Aids in Africa: Planning for a long war',
     'Mars rover maneuvers for rim shot',
     'Mars express confirms presence of water at Mars south pole',
     'NASA announces major Mars rover finding',
     'Drug access, Asia threat in focus at AIDS summit',
     'NASA Mars Odyssey THEMIS image: typical crater',
     'Road blocks for Aids walter goulet optimus prime'],
     'YEAR':['2017', '2018', '2017', '2017', '2018', '2018', '2018'],
     'ID':[1,2,3,4,5,6,7]}))
    data = dat.split(ratio=(0.7,0.3), seed = 1234)
    train_dat = data[0]
    test_dat = data[1]
    # Specify settings.
    cur = cursor()
    try:
        cur.execute("Begin ctx_ddl.drop_policy('DMDEMO_ESA_POLICY'); End;")
    except:
        pass
    cur.execute("Begin ctx_ddl.create_policy('DMDEMO_ESA_POLICY'); End;")
    cur.close()

    odm_settings = {'odms_text_policy_name': 'DMDEMO_ESA_POLICY',
                    '"ODMS_TEXT_MIN_DOCUMENTS"': 1,
                    '"ESAS_MIN_ITEMS"': 1}

    ctx_settings = {'COMMENTS': 
                    'TEXT(POLICY_NAME:DMDEMO_ESA_POLICY)(TOKEN_TYPE:STEM)'}

    # Create an oml ESA model object.
    esa_mod = oml.esa(**odm_settings)

    # Fit the ESA model according to the training data and parameter settings.
    esa_mod = esa_mod.fit(train_dat, case_id = 'ID', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod

    # Use the model to make predictions on test data.
    esa_mod.predict(test_dat, 
                    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']])

    esa_mod.transform(test_dat, 
    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']], 
                                topN = 2).sort_values(by = ['ID'])

    esa_mod.feature_compare(dat, 
                            compare_cols = 'COMMENTS', 
                            supplemental_cols = ['ID'])

    pprint.pprint(dat)
    analysis = esa_mod.feature_compare(dat,
                            compare_cols = ['COMMENTS'],
                            supplemental_cols = ['ID'])
    res = analysis.sort_values(by = ['SIMILARITY'],ascending=False)
    pprint.pprint(res)
    # Change the setting parameter and refit the model.
    new_setting = {'ESAS_VALUE_THRESHOLD': '0.01', 
                'ODMS_TEXT_MAX_FEATURES': '2', 
                'ESAS_TOPN_FEATURES': '2'}
    esa_mod.set_params(**new_setting).fit(train_dat, 'ID', case_id = 'ID', 
                    ctx_settings = ctx_settings)
    pprint.pprint(esa_mod)

    cur = cursor()
    cur.execute("Begin ctx_ddl.drop_policy('DMDEMO_ESA_POLICY'); End;")
    cur.close()
"""
    # Specify settings.
    cur = cursor()
    cur.execute("Begin ctx_ddl.create_policy('DMDEMO_ESA_POLICY'); End;")
    cur.close()

    odm_settings = {'odms_text_policy_name': 'DMDEMO_ESA_POLICY',
                    '"ODMS_TEXT_MIN_DOCUMENTS"': 1,
                    '"ESAS_MIN_ITEMS"': 1}

    ctx_settings = {'COMMENTS': 
                    'TEXT(POLICY_NAME:DMDEMO_ESA_POLICY)(TOKEN_TYPE:STEM)'}

    # Create an oml ESA model object.
    esa_mod = oml.esa(**odm_settings)

    # Fit the ESA model according to the training data and parameter settings.
    esa_mod = esa_mod.fit(train_dat, case_id = 'chiartinstid', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod

    # Use the model to make predictions on test data.
    esa_mod.predict(test_dat, 
                    supplemental_cols = test_dat[:, ['chiartinstid', 'description']])

    esa_mod.transform(test_dat, 
    supplemental_cols = test_dat[:, ['chiartinstid', 'description']], 
                                topN = 2).sort_values(by = ['chiartinstid'])

    esa_mod.feature_compare(test_dat, 
                            compare_cols = 'description', 
                            supplemental_cols = ['chiartinstid'])

    esa_mod.feature_compare(test_dat,
                            compare_cols = ['description', 'artist_name'],
                            supplemental_cols = ['chiartinstid'])

    # Change the setting parameter and refit the model.
    new_setting = {'ESAS_VALUE_THRESHOLD': '0.01', 
                'ODMS_TEXT_MAX_FEATURES': '2', 
                'ESAS_TOPN_FEATURES': '2'}
    esa_mod.set_params(**new_setting).fit(train_dat, 'chiartinstid', case_id = 'chiartinstid', 
                    ctx_settings = ctx_settings)

    cur = cursor()
    cur.execute("Begin ctx_ddl.drop_policy('DMDEMO_ESA_POLICY'); End;")
    cur.close()
"""
    
        

if __name__ == "__main__":
    main(sys.argv)
