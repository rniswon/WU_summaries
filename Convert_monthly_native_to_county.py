import pandas as pd
import numpy as np
import os
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def ps_cu_wsa_data(df):
    """Pivot wsa data and perform necessary transformations."""
    # Convert wsa to a string
    #df['wsa_agidf'] = [str(int(v)).zfill(12) for v in df['wsa_agidf']]
    # Ensure no leading/trailing spaces in column names
    df.columns = df.columns.str.strip()
    # Create a PeriodIndex for monthly data
    df['period'] = pd.PeriodIndex(year=df['year'], month=df['month'], freq='M')
    df.set_index('period', inplace=True)
    df['days_in_month'] = df.index.days_in_month
    # Convert units
    df['ps_cu_m3_mtot'] = df['ps_cu_m3_mtot'] / 3785.41178
    df['ps_cu_m3_mtot'] = df['ps_cu_m3_mtot'] / df['days_in_month']
    del df['days_in_month']
    # Pivot the DataFrame
    #df_pivoted = df.pivot_table(index=['year', 'month'], columns='wsa_agidf', values='ps_cu_m3_mtot')
    # Reset the index to make 'year' and 'month' columns again
    #df_pivoted.reset_index(inplace=True)
    # Flatten the columns for clarity
    #df_pivoted.columns.name = None
    return df
def year_pivot(df):
    # remove decimal and zero from year
    df['year'] = df['year'].astype(int)
    # Pivot for cu_mgd
    cu_mgd_pivot = df.pivot(index='county', columns='year', values='cumgd')
    cu_mgd_pivot.columns = [f'cumgd{col}' for col in cu_mgd_pivot.columns]

    # Pivot for wd_mgd
    wd_mgd_pivot = df.pivot(index='county', columns='year', values='wdmgd')
    wd_mgd_pivot.columns = [f'wdmgd{col}' for col in wd_mgd_pivot.columns]

    # Combine the pivoted dataframes
    df = cu_mgd_pivot.join(wd_mgd_pivot).reset_index()
    return df
def rename_field(gdf, old_field_name, new_field_name):
    # Check if the old field name exists in the GeoDataFrame
    if old_field_name not in gdf.columns:
        raise ValueError(f"Field name '{old_field_name}' not found in the GeoDataFrame.")

    # Rename the column
    gdf = gdf.rename(columns={old_field_name: new_field_name})

    # Return the processed GeoDataFrame
    return gdf
def load_shapefiles(script_dir):
    county_shape_file = os.path.join(script_dir, "county_shape", "county.shp")
    wsa_county_shape_file = os.path.join(script_dir, "wsa_county", "wsa_county.shp")
    gdf_county = gpd.read_file(county_shape_file)
    gdf_wsa_county = gpd.read_file(wsa_county_shape_file)
    # Convert column names to lowercase
    gdf_county.columns = map(str.lower, gdf_county.columns)
    gdf_wsa_county.columns = map(str.lower, gdf_wsa_county.columns)
    # Define the old field name and the new field name
    old_field_name = "name"
    new_field_name = "county"
    # rename column in shapefiles
    gdf_county = rename_field(gdf_county, old_field_name, new_field_name)
    gdf_wsa_county = rename_field(gdf_wsa_county, old_field_name, new_field_name)
    return gdf_county, gdf_wsa_county
def fast_read_csv(fn):
    """Read CSV file quickly and return as DataFrame, handling quoted values and preserving HUC12 as a string."""
    with open(fn, 'r') as fidr:
        content = fidr.readlines()
    data = []
    i = 0
    huc12_index = None
    for line in tqdm(content):
        if i == 0:
            # Read header
            line = line.strip()
            columns = line.split(",")
            columns = [col.lower() for col in columns]  # Convert column names to lowercase
            if 'huc12' in columns:
                huc12_index = columns.index('huc12')
            i += 1
            continue
        parts = line.strip().split(",")
        if len(parts) > len(columns):
            for ip, p in enumerate(parts):
                if p.count('"') == 1:
                    parts[ip] = parts[ip] + parts[ip + 1]
                    del parts[ip + 1]

        row = []
        for idx, v in enumerate(parts):
            v = v.replace('"', '').replace("'", '')  # Remove quotes
            if idx == huc12_index:
                # Treat HUC12 as a string without attempting to convert it
                row.append(v)
            else:
                try:
                    # Try to convert to float for other columns
                    row.append(float(v))
                except ValueError:
                    # If conversion fails, append the original string
                    row.append(v)
        data.append(row)
        i += 1

    df = pd.DataFrame(data, columns=columns)

    return df

def preprocess_data(file_path, i):
    """Preprocess data from CSV file."""
    df = fast_read_csv(file_path)
    df.replace([999, 888], np.NaN, inplace=True)
    # Remove quotes from headers
    df.columns = df.columns.str.replace('"', '')
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    return df

def ir_merge_and_sum_by_county(script_dir, df):
    """load IR HUC12 to county crosswalk"""
    file_name="IR_huc12_to_us_counties.csv"
    file_path=os.path.join(script_dir, file_name)
    df_crosswalk = fast_read_csv(file_path)
    df = pd.melt(df, id_vars=['year', 'month'], var_name='huc12', value_name='wd_mgd')
    df['period'] = pd.PeriodIndex(year=df['year'], month=df['month'], freq='M')
    df.set_index('period', inplace=True)
    df['days_in_month'] = df.index.days_in_month
    df['wd_mgd'] = df['wd_mgd']*df['days_in_month']
    df = df.groupby(['year', 'huc12']).agg({
        'wd_mgd': 'sum',  # Sum the 'wd_gpd' values
        'days_in_month': 'first'  # Keep the first value of 'days_in_month' for each group
    }).reset_index()
    df['wd_mgd'] = df['wd_mgd'] / df['days_in_month']
    merged_df = df_crosswalk.merge(df, on='huc12', how='inner')
    merged_df.replace([999, 888], np.NaN, inplace=True)
    # Columns to keep
    for year in range(2000, 2021):
        frac_column = f"frac_area_{year}"
        wd_column = f"wd_mgd_{year}"

        # Multiply the fraction column by the wd_mgd column
        merged_df[wd_column] = merged_df[frac_column] * merged_df['wd_mgd']

        # Drop the original fraction column
        merged_df.drop(columns=[frac_column], inplace=True)
    merged_df['county_name'] = merged_df['county_name'].astype(str)
    merged_df['county_geoid'] = merged_df['county_geoid'].astype(str)
    # Order the rows by ascending order of the 'year' column
    merged_df = merged_df.groupby(['county_geoid', 'county_name']).agg({
        'wd_mgd_2000': 'sum',
        'wd_mgd_2001': 'sum',
        'wd_mgd_2002': 'sum',
        'wd_mgd_2003': 'sum',
        'wd_mgd_2004': 'sum',
        'wd_mgd_2005': 'sum',
        'wd_mgd_2006': 'sum',
        'wd_mgd_2007': 'sum',
        'wd_mgd_2008': 'sum',
        'wd_mgd_2009': 'sum',
        'wd_mgd_2010': 'sum',
        'wd_mgd_2011': 'sum',
        'wd_mgd_2012': 'sum',
        'wd_mgd_2013': 'sum',
        'wd_mgd_2014': 'sum',
        'wd_mgd_2015': 'sum',
        'wd_mgd_2016': 'sum',
        'wd_mgd_2017': 'sum',
        'wd_mgd_2018': 'sum',
        'wd_mgd_2019': 'sum',
        'wd_mgd_2020': 'sum'
    }).reset_index()
    merged_df = merged_df.rename(columns={'county_name': 'county'})
    return merged_df
def te_merge_and_sum_by_county(gdf_county, df):
    """Merge data with county GeoDataFrame and sum by state."""
    merged_df = gdf_county.merge(pd.merge(gdf_county, df, on='county', how='inner'))
    # Columns to keep
    columns_to_keep = ['year', 'county', 'state', 'cu_mgd', 'wd_mgd']
    merged_df = merged_df.loc[:, columns_to_keep]
    # Group by both 'county' and 'year' and sum the appropriate columns
    merged_df = merged_df.groupby(['county', 'year']).agg({
        'state': 'first',  # Assuming you want to keep the first state within each group
        'cu_mgd': 'sum',
        'wd_mgd': 'sum',
    }).reset_index()
    merged_df = merged_df.rename(columns={'cu_mgd': 'cumgd'})
    merged_df = merged_df.rename(columns={'wd_mgd': 'wdmgd'})
    # Order the rows by ascending order of the 'year' column
    merged_df = merged_df.sort_values(by=['year', 'county']).reset_index(drop=True)
    return merged_df
def ps_merge_and_sum_by_county(gdf_wsa_county, df):
    df = df.rename(columns={'sys_id': 'wsa_agidf'})
    #"""Merge PS data by county ."""
    merged_df = gdf_wsa_county.merge(pd.merge(gdf_wsa_county, df, on='wsa_agidf', how='inner'))
    # Calc total withdrawal from per capita, population, and area fraction in county
    merged_df['wdgpd']=merged_df['pop']*merged_df['est_per_capita']*merged_df['fraction']
    merged_df['pop'] = merged_df['pop'] * merged_df['fraction']
    merged_df = merged_df.rename(columns={'state_name': 'state'})
    # Columns to keep
    columns_to_keep = ['year', 'county', 'state', 'wdgpd']
    merged_df = merged_df.loc[:, columns_to_keep]
    merged_df['year'] = merged_df['year'].astype(int)
    # Group by both 'county' and 'year' and sum the appropriate columns
    merged_df = merged_df.groupby(['county', 'year']).agg({
        'state': 'first',  # Assuming you want to keep the first state within each group
        'wdgpd': 'sum',
    }).reset_index()
    # Order the rows by ascending order of the 'year' column
    merged_df = merged_df.sort_values(by=['county', 'state']).reset_index(drop=True)
    # Pivot for wd_mgd
    # Pivot the DataFrame
    merged_df = merged_df.pivot(index=['county', 'state'], columns='year', values='wdgpd')
    # Flatten the columns and rename them
    merged_df.columns = [f'wdgpd_{col}' for col in merged_df.columns]
    # Reset the index to make 'county' and 'state' columns again
    merged_df = merged_df.reset_index()
    return merged_df

def ps_cu_merge_and_sum_by_county(gdf_wsa_county, df):
    df = df.rename(columns={'ps_cu_m3_mtot': 'cu_mgd'})
    #"""Merge PS data by county ."""
    merged_df = gdf_wsa_county.merge(pd.merge(gdf_wsa_county, df, on='wsa_agidf', how='inner'))
    merged_df = merged_df.rename(columns={'state_name': 'state'})
    # Columns to keep
    columns_to_keep = ['year', 'county', 'state', 'cu_mgd']
    merged_df = merged_df.loc[:, columns_to_keep]
    merged_df['year'] = merged_df['year'].astype(int)
    # Group by both 'county' and 'year' and sum the appropriate columns
    merged_df = merged_df.groupby(['county', 'year']).agg({
        'state': 'first',  # Assuming you want to keep the first state within each group
        'cu_mgd': 'sum',
    }).reset_index()
    # Order the rows by ascending order of the 'year' column
    merged_df = merged_df.sort_values(by=['county', 'state']).reset_index(drop=True)
    # Pivot for wd_mgd
    # Pivot the DataFrame
    merged_df = merged_df.pivot(index=['county', 'state'], columns='year', values='cu_mgd')
    # Flatten the columns and rename them
    merged_df.columns = [f'cu_mgd_{col}' for col in merged_df.columns]
    # Reset the index to make 'county' and 'state' columns again
    merged_df = merged_df.reset_index()
    return merged_df

def process_file(args):
    """Process individual file."""
    file_path, file_name_out, file_shapes_out, script_dir, gdf_county, gdf_wsa_county, i = args
    df = preprocess_data(file_path, i)
    if i==0:
        merged_df = te_merge_and_sum_by_county(gdf_county, df)
        merged_df = year_pivot(merged_df)
    if i==1:
        merged_df = ps_merge_and_sum_by_county(gdf_wsa_county, df)
    if i==2:
        merged_df = ps_cu_wsa_data(df)
        merged_df = ps_cu_merge_and_sum_by_county(gdf_wsa_county, merged_df)
    if i==3:
        merged_df = ir_merge_and_sum_by_county(script_dir, df)
    # Save the final CSV
        csv_out_path = os.path.join(script_dir, file_name_out)
        merged_df.to_csv(csv_out_path, index=False)
    # Save the final shapefile
        dfs_shape = gdf_county.merge(merged_df, on='county', how='left')
        shape_out_path = os.path.join(script_dir, file_shapes_out)
        dfs_shape.to_file(shape_out_path)

def main():
    """Main function to process all files."""
    script_dir = os.path.dirname(__file__)
    # Define file names and output paths
    file_names = [
        "published_annual_thermoelectric_water_use_estimates_2008-2020.csv",
        "PS_WSA_annual_Withdrawals_2000_2020.csv",
        "ps_mon_CU_SA.csv",
        "IR_HUC12_Tot_WD_monthly_2000_2020.csv"
    ]
    file_names_out = [
        "TE_county_annual_water_use_estimates_2008-2020.csv",
        "PS_county_annual_Withdrawals_2000_2020.csv",
        "PS_county_annual_consumption.csv",
        "IR_county_Tot_WD_annual_2000_2020.csv"
    ]
    file_shapes_out = [
        "shapes/TE_county_annual_Withdrawals_2008-2020.shp",
        "shapes/PS_county_annual_Withdrawals_2000_2020.shp",
        "shapes/ps_mon_CU_SA.shp",
        "IR_county_Tot_WD_annual_2000_2020.shp"
    ]
    # Load shapefiles
    gdf_county, gdf_wsa_county = load_shapefiles(script_dir)
    # Process each file
    for i, file_name in enumerate(file_names):
        args = (os.path.join(script_dir, file_name), file_names_out[i], file_shapes_out[i], script_dir, gdf_county, gdf_wsa_county, i)
        process_file(args)

if __name__ == "__main__":
    main()