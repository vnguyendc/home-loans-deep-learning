import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


#SAMPLE_SIZE = 40000

def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
    """
    Process the input dataframes into a single one containing all the features. Requires
    a lot of aggregating of the supplementary datasets such that they have an entry per
    customer.

    Also, add any new features created from the existing ones
    """

    # # Add new features

    # Amount loaned relative to salary
    app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']

    # # Aggregate and merge supplementary datasets
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))

    # Previous applications
    agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
    prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
    prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
    merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

    # Average the rest of the previous app data
    prev_apps_avg = prev_app_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(prev_apps_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_PAVG'])
    print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

    # Previous app categorical features
    prev_app_df, cat_feats, _ = process_dataframe(prev_app_df)
    prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')\
                             .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
    merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

    # Credit card data - numerical features
    wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
    credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)
    merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CCAVG'])

    # Credit card data - categorical features
    most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CCAVG'])
    print('Shape after merging with credit card data = {}'.format(merged_df.shape))

    # Credit bureau data - numerical features
    credit_bureau_avgs = bureau_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(credit_bureau_avgs, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
    print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

    # Bureau balance data
    most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
    merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                            how='left', suffixes=['', '_B_B'])
    print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

    # Pos cash data - weight values by recency when averaging
    wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
    cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CAVG'])

    # Pos cash data data - categorical features
    most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                       how='left', suffixes=['', '_CAVG'])
    print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

    # Installments data
    ins_avg = install_df.groupby('SK_ID_CURR').mean()
    merged_df = merged_df.merge(ins_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_IAVG'])
    print('Shape after merging with installments data = {}'.format(merged_df.shape))

    # Add more value counts
    merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
    merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
    merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
    merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR',
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    print('Shape after merging with counts data = {}'.format(merged_df.shape))

    return merged_df

def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    categorical_feats = categorical_feats
    encoder_dict = {}
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
        encoder_dict[feat] = encoder

    return input_df, categorical_feats.tolist(), encoder_dict

def main():
    # applications
    try:
        with ZipFile('./data/kaggle/application_train.csv.zip') as zf:
            with zf.open('application_train.csv') as myZip:
                apps_df = pd.read_csv(myZip)
        print('Applications Training data shape: ', apps_df.shape)

        # applications test
        # with ZipFile('application_test.csv.zip') as zf:
        #     with zf.open('application_test.csv') as myZip:
        #         apps_test_df = pd.read_csv(myZip).head(sample_size)

        # bureau
        with ZipFile('./data/kaggle/bureau.csv.zip') as zf:
            with zf.open('bureau.csv') as myZip:
                bureau_df = pd.read_csv(myZip)
        print('Bureau data shape: ', bureau_df.shape)

        # bureau balance
        with ZipFile('./data/kaggle/bureau_balance.csv.zip') as zf:
            with zf.open('bureau_balance.csv') as myZip:
                bureau_balance_df = pd.read_csv(myZip)
        print('Bureau Balance shape: ', bureau_balance_df.shape)

        # credit card balance
        with ZipFile('./data/kaggle/credit_card_balance.csv.zip') as zf:
            with zf.open('credit_card_balance.csv') as myZip:
                credit_card_df = pd.read_csv(myZip)
        print('Credit Card Balance shape: ', credit_card_df.shape)

        # POS CASH
        with ZipFile('./data/kaggle/POS_CASH_balance.csv.zip') as zf:
            with zf.open('POS_CASH_balance.csv') as myZip:
                pos_cash_df = pd.read_csv(myZip)
        print('POS Cash data shape: ', pos_cash_df.shape)

        # previous applications
        with ZipFile('./data/previous_application.csv.zip') as zf:
            with zf.open('previous_application.csv') as myZip:
                prev_apps_df = pd.read_csv(myZip)
        print('Previous applications data shape: ', prev_apps_df.shape)

        # installments payments
        with ZipFile('./data/installments_payments.csv.zip') as zf:
            with zf.open('installments_payments.csv') as myZip:
                install_payments_df = pd.read_csv(myZip)
        print('Installments payments shape: ', install_payments_df.shape)

    except:
        with ZipFile('./data/kaggle/home-credit-default-risk.zip') as zf:
            app = zf.open('application_train.csv')
            bureau = zf.open('bureau.csv')
            bureau_balance = zf.open('bureau_balance.csv')
            credit_card = zf.open('credit_card_balance.csv')
            pos_cash = zf.open('POS_CASH_balance.csv')
            prev_apps = zf.open('previous_application.csv')
            install_payments = zf.open('installments_payments.csv')

            apps_df = pd.read_csv(app)
            print('Applications Training data shape: ', apps_df.shape)

            bureau_df = pd.read_csv(bureau)
            print('Bureau data shape: ', bureau_df.shape)

            bureau_balance_df = pd.read_csv(bureau_balance)
            print('Bureau Balance shape: ', bureau_balance_df.shape)

            credit_card_df = pd.read_csv(credit_card)
            print('Credit Card Balance shape: ', credit_card_df.shape)

            pos_cash_df = pd.read_csv(pos_cash)
            print('POS Cash data shape: ', pos_cash_df.shape)

            prev_apps_df = pd.read_csv(prev_apps)
            print('Previous applications data shape: ', prev_apps_df.shape)

            install_payments_df = pd.read_csv(install_payments)
            print('Installments payments shape: ', install_payments_df.shape)


    # Merge the datasets into a single one for training
    len_train = len(apps_df)
    #app_both = pd.concat([apps_df, apps_test_df])
    merged_df = feature_engineering(apps_df, bureau_df, bureau_balance_df, credit_card_df,
                                    pos_cash_df, prev_apps_df, install_payments_df)

    # Separate metadata
    meta_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']
    meta_df = merged_df[meta_cols]
    merged_df.drop(meta_cols, axis=1, inplace=True)

    # Process the data set.
    merged_df, categorical_feats, encoder_dict = process_dataframe(input_df=merged_df)

    # Capture other categorical features not as object data types:
    non_obj_categoricals = [
        'FONDKAPREMONT_MODE',
        'HOUR_APPR_PROCESS_START',
        'HOUSETYPE_MODE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE',
        'NAME_TYPE_SUITE',
        'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE',
        'WALLSMATERIAL_MODE',
        'WEEKDAY_APPR_PROCESS_START',
        'NAME_CONTRACT_TYPE_BAVG',
        'WEEKDAY_APPR_PROCESS_START_BAVG',
        'NAME_CASH_LOAN_PURPOSE',
        'NAME_CONTRACT_STATUS',
        'NAME_PAYMENT_TYPE',
        'CODE_REJECT_REASON',
        'NAME_TYPE_SUITE_BAVG',
        'NAME_CLIENT_TYPE',
        'NAME_GOODS_CATEGORY',
        'NAME_PORTFOLIO',
        'NAME_PRODUCT_TYPE',
        'CHANNEL_TYPE',
        'NAME_SELLER_INDUSTRY',
        'NAME_YIELD_GROUP',
        'PRODUCT_COMBINATION',
        'NAME_CONTRACT_STATUS_CCAVG',
        'STATUS',
        'NAME_CONTRACT_STATUS_CAVG'
    ]
    categorical_feats = categorical_feats + non_obj_categoricals

    # convert float non object categoricals to int
    merged_df[non_obj_categoricals] = merged_df[non_obj_categoricals].astype(int)

    # Extract target before scaling
    labels = merged_df.pop('TARGET')
    labels = labels[:len_train]

    # Reshape (one-hot)
    target = np.zeros([len(labels), len(np.unique(labels))])
    target[:, 0] = labels == 0
    target[:, 1] = labels == 1

    ## check nulls
    null_counts = merged_df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    null_ratios = null_counts / len(merged_df)

    # Drop columns over x% null
    null_thresh = .8
    null_cols = null_ratios[null_ratios > null_thresh].index
    merged_df.drop(null_cols, axis=1, inplace=True)
    print('Columns dropped for being over {}% null:'.format(100*null_thresh))
    for col in null_cols:
        print(col)
        if col in categorical_feats:
            categorical_feats.pop(col)

    # Fill the rest with the mean (TODO: do something better!)
    # merged_df.fillna(merged_df.median(), inplace=True)
    merged_df.fillna(0, inplace=True)

    ## identify categoricals
    cat_feats_idx = np.array([merged_df.columns.get_loc(x) for x in categorical_feats])
    int_feats_idx = [merged_df.columns.get_loc(x) for x in non_obj_categoricals]
    cat_feat_lookup = pd.DataFrame({'feature': categorical_feats, 'column_index': cat_feats_idx})

    ## identify continuous
    cont_feats_idx = np.array(
        [merged_df.columns.get_loc(x)
        for x in merged_df.columns[~merged_df.columns.isin(categorical_feats)]]
    )
    cont_feat_lookup = pd.DataFrame(
        {'feature': merged_df.columns[~merged_df.columns.isin(categorical_feats)],
        'column_index': cont_feats_idx}
    )

    # change to allowed scope names
    cont_feat_lookup.feature = cont_feat_lookup.feature.apply(lambda x: x.replace(' ', '_'))
    cat_feat_lookup.feature = cat_feat_lookup.feature.apply(lambda x: x.replace(' ', '_'))

    ## scaling
    scaler = StandardScaler()
    final_col_names = merged_df.columns
    merged_df = merged_df.values
    merged_df[:, cont_feats_idx] = scaler.fit_transform(merged_df[:, cont_feats_idx])

    scaler_2 = MinMaxScaler(feature_range=(0, 1))
    merged_df[:, int_feats_idx] = scaler_2.fit_transform(merged_df[:, int_feats_idx])


    ### save to data folder
    cont_feat_lookup.to_csv('./data/cont_feat_lookup.csv', index=False)
    cat_feat_lookup.to_csv('./data/cat_feat_lookup.csv', index=False)
    pd.DataFrame(merged_df, columns=final_col_names).to_csv('./data/train/home_loans_train.csv', index=False)
    pd.DataFrame(target).to_csv('./data/train/home_loans_target.csv', index=False)

##### MAIN #####
main()