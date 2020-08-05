import pandas as pd

def add_numberid(logparser_templates_file):
    df = pd.read_csv(logparser_templates_file, header=0)
    df['numberID'] = range(1, len(df) + 1)
    print(df)

    df.to_csv(logparser_templates_file, columns=df.columns, index=0, header=1)


def add_numberid_new(logparser_templates_file):
    df = pd.read_csv(logparser_templates_file, header=0)
    df['numberID'] = ['E' + str(x) for x in range(1, len(df) + 1)]
    print(df)

    df.to_csv(logparser_templates_file, columns=df.columns, index=0, header=1)

