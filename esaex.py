import oml
from oml import cursor
import pandas as pd
import sys
import os
import pprint

def main(argv):
    omlusr = os.environ.get("OMLUSERNAME")
    omlpass = os.environ.get("OMLPASS")
    oml.connect(user=omlusr,password=omlpass,dsn="aidb_medium",automl="aidb_medium_pool")
    # Create training data and test data.
    dat = oml.push(pd.DataFrame( 
    {'COMMENTS':['Aids in Africa: Planning for a long war',
     'Mars rover maneuvers for rim shot',
     'Mars express confirms presence of water at Mars south pole',
     'NASA announces major Mars rover finding',
     'Drug access, Asia threat in focus at AIDS summit',
     'NASA Mars Odyssey THEMIS image: typical crater',
     'Road blocks for Aids'],
     'YEAR':['2017', '2018', '2017', '2017', '2018', '2018', '2018'],
     'ID':[1,2,3,4,5,6,7]}))

    datadf = dat.pull()

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

    ctx_settings = {'COMMENTS': 
                    'TEXT(POLICY_NAME:DEMO_ESA_POLICYV1)(TOKEN_TYPE:STEM)'}

    # Create an oml ESA model object.
    esa_mod = oml.esa(**odm_settings)

    # Fit the ESA model according to the data and parameter settings.
    esa_mod = esa_mod.fit(dat, case_id = 'ID', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod

    esa_mod.transform(dat, 
    supplemental_cols = dat[:, ['ID', 'COMMENTS']], 
                                topN = 2).sort_values(by = ['ID'])

    results = esa_mod.feature_compare(dat, 
                            compare_cols = 'COMMENTS', 
                            supplemental_cols = ['ID'])
    datadf = dat.pull()
    print(datadf)
    print(results)
    resultdf = results.sort_values(by = ['SIMILARITY'],ascending=False).head(1).pull()
    sim = resultdf.loc[0]['SIMILARITY']
    idA = resultdf.loc[0]['ID_A']
    idB = resultdf.loc[0]['ID_B']
    print(f"With highest correlation {sim} entries with index {idA} and {idB} are most similar.")
    print(f"Record A: {datadf.loc[datadf.ID == idA,'COMMENTS'].values[0]}")
    print(f"Record B: {datadf.loc[datadf.ID == idB,'COMMENTS'].values[0]}")


if __name__ == "__main__":
    main(sys.argv)
