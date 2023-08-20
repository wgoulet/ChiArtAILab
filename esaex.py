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
     'ID':[1,2,3,4,5,6,7]})).split(ratio=(0.7,0.3),seed = 1234)

    train_dat = dat[0]
    test_dat = dat[1]
    all_dat = dat[0].append(dat[1])
    outfile = 'log.txt'
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(all_dat))
        

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
    esa_mod = esa_mod.fit(train_dat, case_id = 'ID', 
                        ctx_settings = ctx_settings)

    # Show model details.
    esa_mod
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(esa_mod))
        f.write("\n") 

    # Use the model to make predictions on test data.
    w = esa_mod.predict(test_dat, 
                    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']])
    trainfeatures = esa_mod.features.pull()
    predictions = w.pull()
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(predictions))
        f.write("\n") 
    predfeature = []
    for i in predictions.index:
        fid = predictions.loc[i]['FEATURE_ID']
        predfeature.append(predictions.loc[i]['COMMENTS'])
        for v in trainfeatures.loc[trainfeatures.FEATURE_ID == fid,'ATTRIBUTE_NAME'].values:
            predfeature.append(v)
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(predfeature))
        f.write("\n") 
    esa_mod.transform(test_dat, 
    supplemental_cols = test_dat[:, ['ID', 'COMMENTS']], 
                                topN = 2).sort_values(by = ['ID'])

    x = esa_mod.feature_compare(all_dat, 
                            compare_cols = 'COMMENTS', 
                            supplemental_cols = ['ID'])
    x = x.sort_values(by = ['SIMILARITY'],ascending=False).head(5).pull()
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(x))
        f.write("\n") 
    y = esa_mod.feature_compare(all_dat,
                            compare_cols = ['COMMENTS', 'YEAR'],
                            supplemental_cols = ['ID'])
    y = y.sort_values(by = ['SIMILARITY'],ascending=False).head(5).pull()
    with open(outfile,"+a") as f:
        f.write(pprint.pformat(y))
        f.write("\n") 

    # Change the setting parameter and refit the model.
    new_setting = {'ESAS_VALUE_THRESHOLD': '0.01', 
                'ODMS_TEXT_MAX_FEATURES': '2', 
                'ESAS_TOPN_FEATURES': '2'}
    esa_mod.set_params(**new_setting).fit(train_dat, 'ID', case_id = 'ID', 
                    ctx_settings = ctx_settings)

    cur = cursor()
    cur.execute("Begin ctx_ddl.drop_policy('DEMO_ESA_POLICYV1'); End;")
    cur.close()

if __name__ == "__main__":
    main(sys.argv)
