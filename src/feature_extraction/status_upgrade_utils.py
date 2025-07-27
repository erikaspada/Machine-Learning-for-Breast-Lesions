def assign_status(df, positive_ids):
    df["Status"] = df["PatientID"].apply(lambda x: 1 if x in positive_ids else 0)
    return df

def assign_upgrade(df, upgrade_ids):
    df["Upgrade"] = df["PatientID"].apply(lambda x: 1 if x in upgrade_ids else 0)
    return df
