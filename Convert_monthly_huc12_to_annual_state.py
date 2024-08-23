import pandas as pd
import numpy as np
import os
import geopandas as gpd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def prepend_year_to_numeric_strings(col):
    """Prepend 'year' to numeric string columns."""
    return f"year{col}" if col.isdigit() else col

def pivot_huc12_data(df):
    """Pivot HUC12 data and perform necessary transformations."""
    # Convert HUC12 to a string with leading zeros
    df['huc12'] = [str(int(v)).zfill(12) for v in df['huc12']]
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
    df_pivoted = df.pivot_table(index=['year', 'month'], columns='huc12', values='ps_cu_m3_mtot')
    # Reset the index to make 'year' and 'month' columns again
    df_pivoted.reset_index(inplace=True)
    # Flatten the columns for clarity
    df_pivoted.columns.name = None
    return df_pivoted

def load_shapefiles(script_dir):
    """Load state and HUC12 shapefiles."""
    state_shape_file = os.path.join(script_dir, "state_shape", "cb_2018_us_state_500k.shp")
    huc12_shape_file = os.path.join(script_dir, "huc12_to_states", "huc12_to_states.shp")
    gdf_state = gpd.read_file(state_shape_file)
    gdf_huc12 = gpd.read_file(huc12_shape_file)
    # Convert column names to lowercase
    gdf_state.columns = map(str.lower, gdf_state.columns)
    gdf_huc12.columns = map(str.lower, gdf_huc12.columns)
    return gdf_state, gdf_huc12

def fast_read_csv(fn):
    """Read CSV file quickly and return as DataFrame, handling quoted values and skipping non-numeric values."""
    with open(fn, 'r') as fidr:
        content = fidr.readlines()
    data = []
    i = 0
    for line in tqdm(content):
        if i == 0:
            # Read header
            line = line.strip()
            columns = line.split(",")
            i += 1
            continue
        parts = line.strip().split(",")
        row = []
        for v in parts:
            v = v.replace('"', '').replace("'", '')  # Remove quotes
            try:
                # Try to convert to float
                row.append(float(v))
            except ValueError:
                # If conversion fails, append the original string
                row.append(v)
        data.append(row)
        i += 1
    df = pd.DataFrame(data, columns=columns)
    df.columns = map(str.lower, df.columns)
    return df

def preprocess_data(file_path, i):
    """Preprocess data from CSV file."""
    df = fast_read_csv(file_path)
# special case for PS CU
    if i == 8:
        df = pivot_huc12_data(df)
    # Replace sentinel values with NaN
    df.replace([999, 888], np.NaN, inplace=True)
    # Remove quotes from headers
    df.columns = df.columns.str.replace('"', '')
    # Create a PeriodIndex for monthly data
    df['period'] = pd.PeriodIndex(year=df['year'], month=df['month'], freq='M')
    df.set_index('period', inplace=True)
    df['days_in_month'] = df.index.days_in_month
    # Adjust values by the number of days in the month
    for col in tqdm(df.columns):
        if col in ['year', 'month', 'period', 'days_in_month']:
            continue
        df[col] *= df["days_in_month"]
    # Group by year and sum the data
    df = df.groupby('year').sum()
    for col in tqdm(df.columns):
        if col in ['year', 'month', 'period', 'days_in_month']:
            continue
        df[col] /= df['days_in_month']
        df[col] = df[col].round(4)
    # Drop unnecessary columns and transpose the DataFrame
    df.drop(columns=['month', 'days_in_month'], inplace=True)
    df = df.transpose().reset_index()
    df.rename(columns={'index': 'huc12'}, inplace=True)
    return df

def merge_and_sum_by_state(gdf_huc12, df):
    """Merge data with HUC12 GeoDataFrame and sum by state."""
    df['huc12'] = df['huc12'].str.zfill(12)
    merged_df = gdf_huc12.merge(df, on='huc12', how='left')
    # Strip any extra spaces from column names
    merged_df.columns = merged_df.columns.astype(str).str.strip()
    merged_df.columns = map(str.lower, merged_df.columns)
    # Identify year columns starting with "2"
    mask = merged_df.columns.str.startswith("2")
    subset_columns = merged_df.columns[mask]
    # Multiply by fraction and sum by state
    for col in subset_columns:
        merged_df[col] *= merged_df['frac2']
    merged_df = merged_df.groupby('stusps').sum().reset_index()
    return merged_df

def extract_data_of_interest(merged_df):
    """Extract and rename data of interest from the merged DataFrame."""
    # Identify columns for years starting with "2"
    mask = merged_df.columns.str.startswith("2")
    subset_columns = merged_df.columns[mask]
    dfs_final = pd.DataFrame(columns=subset_columns)
    dfs_final = merged_df[['stusps']].copy()
    for col in subset_columns:
        dfs_final[col] = merged_df[col]
    # Rename columns to prepend 'year' to numeric strings
    dfs_final.columns = [prepend_year_to_numeric_strings(col) for col in dfs_final.columns]
    return dfs_final

def process_file(args):
    """Process individual file."""
    file_path, file_name_out, file_shapes_out, script_dir, gdf_state, gdf_huc12, i = args
    df = preprocess_data(file_path, i)
    merged_df = merge_and_sum_by_state(gdf_huc12, df)
    dfs_final = extract_data_of_interest(merged_df)
    # Save the final CSV
    csv_out_path = os.path.join(script_dir, file_name_out)
    dfs_final.to_csv(csv_out_path, index=False)
    # Save the final shapefile
    dfs_shape = gdf_state.merge(dfs_final, on='stusps', how='left')
    shape_out_path = os.path.join(script_dir, file_shapes_out)
    dfs_shape.to_file(shape_out_path)

def main():
    """Main function to process all files."""
    script_dir = os.path.dirname(__file__)
    # Define file names and output paths
    file_names = [
        "te_cu_mgd_gw_fresh_huc12_aggregation.csv",
        "te_cu_mgd_sw_fresh_huc12_aggregation.csv",
        "te_wd_mgd_gw_fresh_huc12_aggregation.csv",
        "te_wd_mgd_sw_fresh_huc12_aggregation.csv",
        "IR_HUC12_GW_WD_monthly_2000_2020.csv",
        "IR_HUC12_SW_WD_monthly_2000_2020.csv",
        "IR_HUC12_Tot_WD_monthly_2000_2020.csv",
        "IR_HUC12_CU_monthly_2000_2020.csv",
        "ps_mon_CU_ETr.csv",         #this file is a special case (i=8)
        "PS_HUC12_GW_2000_2020.csv",
        "PS_HUC12_SW_2000_2020.csv",
        "PS_HUC12_Tot_2000_2020.csv"
    ]
    file_names_out = [
        "te_cu_mgd_gw_fresh_state_annual.csv",
        "te_cu_mgd_sw_fresh_state_annual.csv",
        "te_wd_mgd_gw_fresh_state_annual.csv",
        "te_wd_mgd_sw_fresh_state_annual.csv",
        "IR_state_GW_WD_annual_2000_2020.csv",
        "IR_state_SW_WD_annual_2000_2020.csv",
        "IR_state_Tot_WD_annual_2000_2020.csv",
        "IR_state_CU_annual_2000_2020.csv",
        "ps_state_CU_ETr.csv",
        "PS_state_GW_2000_2020.csv",
        "PS_state_SW_2000_2020.csv",
        "PS_state_Tot_2000_2020.csv"
    ]
    file_shapes_out = [
        "shapes/te_cu_mgd_gw_fresh_state_annual.shp",
        "shapes/te_cu_mgd_sw_fresh_state_annual.shp",
        "shapes/te_wd_mgd_gw_fresh_state_annual.shp",
        "shapes/te_wd_mgd_sw_fresh_state_annual.shp",
        "shapes/IR_state_GW_WD_annual_2000_2020.shp",
        "shapes/IR_state_SW_WD_annual_2000_2020.shp",
        "shapes/IR_state_Tot_WD_annual_2000_2020.shp",
        "shapes/IR_state_CU_annual_2000_2020.shp",
        "shapes/ps_state_CU_ETr.shp",
        "shapes/PS_state_GW_2000_2020.shp",
        "shapes/PS_state_SW_2000_2020.shp",
        "shapes/PS_state_Tot_2000_2020.shp"
    ]
    # Load shapefiles
    gdf_state, gdf_huc12 = load_shapefiles(script_dir)
    # Process each file
    for i, file_name in enumerate(file_names):
        args = (os.path.join(script_dir, file_name), file_names_out[i], file_shapes_out[i], script_dir, gdf_state, gdf_huc12, i)
        process_file(args)

if __name__ == "__main__":
    main()