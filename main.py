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
    chiartdata = oml.sync(table='CHIARTDATA').head(20)
    # The column names in the data have to be in uppercase apparently; otherwise the esa model
    # won't load correctly!
    chiartdata = chiartdata.rename({'description':'DESCRIPTION','artist_name':'ARTIST_NAME','chiartinstid':'CHIARTINSTID'}) 
    print(chiartdata)
    datadf = chiartdata.pull()

    # Specify settings.
    cur = cursor()
    try:
        # Cleanup old policy and model from DB if it exists
        cur.execute("Begin ctx_ddl.drop_policy('DEMO_ESA_POLICYV1'); End;")
        oml.drop(model='ID')
    except:
        pass
    cur.execute("Begin ctx_ddl.create_policy('DEMO_ESA_POLICYV1'); End;")
    cur.close()

    odm_settings = {'odms_text_policy_name': 'DEMO_ESA_POLICYV1',
                    '"ODMS_TEXT_MIN_DOCUMENTS"': 1,
                    '"ESAS_MIN_ITEMS"': 1}

    ctx_settings = {'description': 
                    'TEXT(POLICY_NAME:DEMO_ESA_POLICYV1)(TOKEN_TYPE:STEM)'}

    # Create an oml ESA model object.
    esa_mod = oml.esa(**odm_settings)

    # Fit the ESA model according to the data and parameter settings.
    # Bizarre behavior here; the model fails to load if the column names in the data are lowercased
    esa_mod = esa_mod.fit(chiartdata, case_id = 'CHIARTINSTID', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod

    esa_mod.transform(chiartdata, 
    supplemental_cols = chiartdata[:, ['CHIARTINSTID', 'DESCRIPTION']], 
                                topN = 2).sort_values(by = ['CHIARTINSTID'])

    results = esa_mod.feature_compare(chiartdata, 
                            compare_cols = 'DESCRIPTION', 
                            supplemental_cols = ['CHIARTINSTID'])
    print(results)
    resultdf = results.sort_values(by = ['SIMILARITY'],ascending=False).head(1).pull()
    sim = resultdf.loc[0]['SIMILARITY']
    idA = resultdf.loc[0]['CHIARTINSTID_A']
    idB = resultdf.loc[0]['CHIARTINSTID_B']
    print(f"With highest correlation {sim} entries with index {idA} and {idB} are most similar.")
    print(f"Record A: {datadf.loc[datadf.CHIARTINSTID == idA,'DESCRIPTION'].values[0]}")
    print(f"Record B: {datadf.loc[datadf.CHIARTINSTID == idB,'DESCRIPTION'].values[0]}")


if __name__ == "__main__":
    main(sys.argv)
