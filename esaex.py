import oml
from oml import cursor
import pandas as pd
import sys
import os
import pprint
        #'Mars rover takes Drugs in water from Asia',

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
        'Road blocks for Aids',
        'Drugs and Aids major threats to America'],
        'YEAR':['2017', '2018', '2017', '2017','2020', '2018', '2018', '2018'],
        'ID':[1,2,3,4,5,6,7,8]}))
    datadf = dat.pull()
    data = dat.split(ratio=(0.65,0.35), seed = 1234)
    train_dat = data[0]
    test_dat = data[1]
    # Specify settings.
    cur = cursor()
    try:
        cur.execute("Begin ctx_ddl.drop_policy('DEMO_ESA_POLICY'); End;")
    except:
        pass
    cur.execute("Begin ctx_ddl.create_policy('DEMO_ESA_POLICY'); End;")
    cur.close()

    odm_settings = {'odms_text_policy_name': 'DEMO_ESA_POLICY',
                    '"ODMS_TEXT_MIN_DOCUMENTS"': 1,
                    '"ESAS_MIN_ITEMS"': 1}

    ctx_settings = {'COMMENTS': 
                    'TEXT(POLICY_NAME:DEMO_ESA_POLICY)(TOKEN_TYPE:STEM)'}

    # Create an oml ESA model object.
    esa_mod = oml.esa(**odm_settings)

    # Fit the ESA model according to the training data and parameter settings.
    esa_mod = esa_mod.fit(train_dat, case_id = 'ID', 
                        ctx_settings = ctx_settings)


    # Use the model to make predictions on test data.
    pred = esa_mod.predict(test_dat, 
                    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']])
    df = pred.pull()
    print("Predicting which feature the test data is most likely to align to")
    pprint.pprint(pred)
    print("Done Predicting")

    esa_mod.transform(test_dat, 
    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']], 
                                topN = 2).sort_values(by = ['ID'])

    analysis = esa_mod.feature_compare(test_dat, 
                            compare_cols = 'COMMENTS', 
                            supplemental_cols = ['ID'])
    res = analysis.sort_values(by = ['SIMILARITY'],ascending=False)
    # Get most similar entries and print out the score and the matching lines
    resdf = res.head(1).pull()
    sim = resdf.loc[0]['SIMILARITY']
    idA = resdf.loc[0]['ID_A']
    idB = resdf.loc[0]['ID_B']
    print(f"With highest correlation {sim} entries with index {idA} and {idB} are most similar.")
    print(f"Record A: {datadf.loc[idA]['COMMENTS']}")
    print(f"Record B: {datadf.loc[idB]['COMMENTS']}")

    pprint.pprint(res)

    esa_mod.feature_compare(test_dat,
                            compare_cols = ['COMMENTS', 'YEAR'],
                            supplemental_cols = ['ID'])

    cur = cursor()
    cur.execute("Begin ctx_ddl.drop_policy('DEMO_ESA_POLICY'); End;")
    cur.close()

if __name__ == "__main__":
    main(sys.argv)
