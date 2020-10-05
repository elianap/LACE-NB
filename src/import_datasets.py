#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
warnings.simplefilter('ignore')
from src.dataset import Dataset


def getAttributes(df, ignoreCols=[]):
    columns=df.columns.drop(ignoreCols)
    attributes=[(c, list(df[c].unique().astype(str))) for c in columns]
    return attributes
def importZooDataset(inputDir="./datasets"):
    meta_col="name"
    import pandas as pd
    df=pd.read_csv(f"{inputDir}/zoo.tab", sep="\t", dtype=str)
    import random
    random.seed=7
    explain_indices = list(random.sample(range(len(df)), 100))
    df_train=df
    df_explain=df
    attributes=getAttributes(df, ignoreCols=[meta_col])
    d_train=Dataset(df_train.drop(columns=meta_col).values, attributes, df_train[[meta_col]].values)
    d_explain=Dataset(df_explain.drop(columns=meta_col).values, attributes, df_explain[meta_col].values)
    return d_train, d_explain, [str(i) for i in explain_indices]

#COMPAS
#https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <=0:
        return '0'
    elif 1<=x<=3:
        return '[1-3]'
    else:
        return '>3'

    
# Quantize length of stay
def quantizeLOS(x):
    if x<= 7:
        return '<week'
    if 8<x<=93:
        return '1w-3M'
    else:
        return '>3Months'
    
    
def get_decile_score_class(x):
        if x >=8:
            return 'High'
        else:
            return 'Medium-Low'


def import_process_compas(risk_class=True, inputDir="./datasets"):
    import pandas as pd
    df_raw=pd.read_csv(f"{inputDir}/compas-scores-two-years.csv")
    cols_propb=[ "c_charge_degree", "race", "age_cat", "sex", "priors_count", "days_b_screening_arrest", "two_year_recid"]#, "is_recid"] 
    cols_propb.sort()
    #df_raw[["days_b_screening_arrest"]].describe()
    df=df_raw[cols_propb]
    #Warning
    df['length_of_stay'] = ((pd.to_datetime(df_raw['c_jail_out']).dt.date - pd.to_datetime(df_raw['c_jail_in']).dt.date).dt.days).copy()

    df=df.loc[abs(df["days_b_screening_arrest"])<=30]#.sort_values("days_b_screening_arrest")
    #df=df.loc[df["is_recid"]!=-1]
    df=df.loc[df["c_charge_degree"]!="O"] # F: felony, M: misconduct
    discrete=["age_cat", "c_charge_degree","race","sex","two_year_recid"]#, "is_recid"]
    continuous=["days_b_screening_arrest","priors_count", "length_of_stay"]
    
    df["priors_count"]=df["priors_count"].apply(lambda x: quantizePrior(x))
    df["length_of_stay"]=df["length_of_stay"].apply(lambda x: quantizeLOS(x))
    df=df[discrete+["priors_count", "length_of_stay"]]
    
    if risk_class:
        df["class"]=df_raw['decile_score'].apply(get_decile_score_class) 
        df.drop(columns="two_year_recid", inplace=True)
    else:
        df.rename(columns={"two_year_recid":"class"}, inplace=True)
        
    return df