import sys
import pprint
import requests
import urllib
import time
import math
import pandas as pd
import numpy as np
import oml
import os
import pickle
from sklearn.datasets import load_digits
import logging
import logging.handlers
from oml import cursor
import cx_Oracle
from oml.mlx import GlobalFeatureImportance

def main(argv):
    omlusr = os.environ.get("OMLUSERNAME")
    omlpass = os.environ.get("OMLPASS")
    oml.connect(user=omlusr,password=omlpass,dsn="aidb_medium",automl="aidb_medium_pool")
    chiartdata = oml.sync(table='CHIARTDATA').tail(500)
    # The column names in the data have to be in uppercase apparently; otherwise the esa model
    # won't load correctly!
    chiartdata = chiartdata.rename({'description':'DESCRIPTION','artist_name':'ARTIST_NAME','chiartinstid':'CHIARTINSTID',
                                    'title':'TITLE'}) 
    # drop duplicate descriptions
    chiartdata = chiartdata.drop_duplicates(subset=['DESCRIPTION'])
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

    # Get extracted features
    features = esa_mod.features
    features
    
    try:
        cur = cursor()
        cur.execute("DROP TABLE OMLUSER.CHIARTFEATURES")
        cur.close()
    except:
        pass
    # Strip out the artist name/title attributes from the features list
    fdata = features.pull()
    noan = fdata.loc[fdata['ATTRIBUTE_NAME'] != 'ARTIST_NAME']
    stripped = noan.loc[fdata['ATTRIBUTE_NAME'] != 'TITLE']
    oml_chiartfeatures = oml.create(stripped,table ="CHIARTFEATURES",dbtypes={'ATTRIBUTE_NAME':'VARCHAR2(4000)','ATTRIBUTE_VALUE':'VARCHAR2(4000)',
                                        'FEAUTURE_ID':'NUMBER','COEFFICIENT':'FLOAT(5)'})
    
    esa_mod.transform(chiartdata, 
    supplemental_cols = chiartdata[:, ['CHIARTINSTID', 'DESCRIPTION']], 
                                topN = 2).sort_values(by = ['CHIARTINSTID'])

    results = esa_mod.feature_compare(chiartdata, 
                            compare_cols = 'DESCRIPTION', 
                            supplemental_cols = ['CHIARTINSTID'])
    dataset = []
    resultdf = results.sort_values(by = ['SIMILARITY'],ascending=False).head(500).pull()
    for i in resultdf.index:
        sim = resultdf.loc[i]['SIMILARITY']
        idA = resultdf.loc[i]['CHIARTINSTID_A']
        idB = resultdf.loc[i]['CHIARTINSTID_B']
        df = pd.DataFrame({
            'artist_name_a':[datadf.loc[datadf.CHIARTINSTID == idA,'ARTIST_NAME'].values[0]], 
            'title_a':[datadf.loc[datadf.CHIARTINSTID == idA,'TITLE'].values[0]],
            'description_a':[datadf.loc[datadf.CHIARTINSTID == idA,'DESCRIPTION'].values[0]],
            'artist_name_b':[datadf.loc[datadf.CHIARTINSTID == idB,'ARTIST_NAME'].values[0]],
            'title_b':[datadf.loc[datadf.CHIARTINSTID == idB,'TITLE'].values[0]],
            'description_b':[datadf.loc[datadf.CHIARTINSTID == idB,'DESCRIPTION'].values[0]],
            'similarity':sim})
        dataset.append(df)

    dbset = pd.concat(dataset)
    try:
        cur = cursor()
        cur.execute("DROP TABLE OMLUSER.CHIARTSIMDATA")
        cur.close()
    except:
        pass
    oml_chiartsimdata = oml.create(dbset,table ="CHIARTSIMDATA",dbtypes={'artist_name_a':'VARCHAR2(4000)','title_a':'VARCHAR2(4000)',
                                        'description_a':'VARCHAR2(4000)','artist_name_b':'VARCHAR2(4000)',
                                        'title_b':'VARCHAR2(4000)','description_b':'VARCHAR2(4000)','similarity':'FLOAT(5)'})

if __name__ == "__main__":
    main(sys.argv)
