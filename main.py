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
    print(chiartdata)
    type(chiartdata)
    datadf = chiartdata.head(10).pull()
    print(datadf.columns)
    print(datadf)
    print(datadf.description)
    hprecords = datadf.loc[datadf.artist_name == 'Hiram Powers']
    print(hprecords)

    
    # Pulling data directly in caused errors in model setup, so manually create dataframe with data to debug
    for i in datadf.chiartinstid:
        #print(f"{datadf.loc[datadf.chiartinstid == i,'artist_name'].values[0]} " \
        #    f"{datadf.loc[datadf.chiartinstid == i,'description'].values[0]} "
        #    f"{datadf.loc[datadf.chiartinstid == i,'chiartinstid'].values[0]} ")
        print(f"{datadf.loc[datadf.chiartinstid == i,'chiartinstid'].values[0]}",end=",")

    chiartdata2 = oml.push(pd.DataFrame( 
    {'DESCRIPTION':[
        'No description available',
        'A work made of marble.',
        'A work made of carrera marble.',
        'A work made of marble.',
        'A work made of carrera marble.',
        'A work made of marble.',
        'A work made of oil on canvas, mounted on panel.',
        'A work made of oil on canvas.',
        'A work made of color screenprint on paper.',
        'Painting of figures construction project amid smoky destruction, middle is pink crater.'
     ],
     'ARTIST_NAME':['Pu Hua','Hiram Powers','Hiram Powers','Hiram Powers','Hiram Powers',
                    'Joseph Mozier','Ilya Bolotowsky','Ilya Bolotowsky','Ilya Bolotowsky','Peter Blume'],
     'CHIARTINSTID':[46352,64076,120518,107863,120515,146929,186682,93779,94097,56682]}))
    
    
    datadf = chiartdata.head(10).pull()
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
    esa_mod = esa_mod.fit(chiartdata, case_id = 'CHIARTINSTID', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod

    esa_mod.transform(chiartdata, 
    supplemental_cols = chiartdata[:, ['chiartinstid', 'description']], 
                                topN = 2).sort_values(by = ['chiartinstid'])

    results = esa_mod.feature_compare(chiartdata, 
                            compare_cols = 'description', 
                            supplemental_cols = ['chiartinstid'])
    print(datadf)
    print(results)
    resultdf = results.sort_values(by = ['SIMILARITY'],ascending=False).head(1).pull()
    sim = resultdf.loc[0]['SIMILARITY']
    idA = resultdf.loc[0]['ID_A']
    idB = resultdf.loc[0]['ID_B']
    print(f"With highest correlation {sim} entries with index {idA} and {idB} are most similar.")
    print(f"Record A: {datadf.loc[datadf.chiartinstid == idA,'description'].values[0]}")
    print(f"Record B: {datadf.loc[datadf.chiartinstid == idB,'description'].values[0]}")


if __name__ == "__main__":
    main(sys.argv)
