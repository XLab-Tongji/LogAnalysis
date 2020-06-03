import pandas as pd

def add_numberid(logparser_templates_file):
    df = pd.read_csv(logparser_templates_file, header=0)
    df['numberID'] = range(len(df))
    print(df)

    df.to_csv(logparser_templates_file, columns=df.columns, index=0, header=1)