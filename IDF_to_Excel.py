# Library imports
import pyidf
import pandas as pd
from pyidf.idf import IDF
from pyidf import ValidationLevel
import os
from os import listdir
import numpy as np

pd.set_option('display.max_columns', None)


# Temporary code:
# file = '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ2A_Tampa_90.1_2016/ASHRAE90.1_ApartmentHighRise_STD2016_Tampa.idf'

# Function to extract materials, constructions, and surfaces from an energyplus idf file
def idf_materials_toexcel(file):
    # Import idf file based upon input string.
    pyidf.validation_level = ValidationLevel.no
    idf = IDF(file)
    file_name = str(os.path.splitext(file)[0]) + str('.xlsx')
    building_name = idf['Building'][0]['Name']

    # Import LCI Data
    LCI_df = pd.read_csv('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/LCI.csv')
    LCI_df = LCI_df[['Name', 'EEC', 'ECC', 'Assumption']]

    # Import LCI Data for Windows
    LCI_window_df = pd.read_csv('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/LCI_Window.csv')
    LCI_window_df = LCI_window_df[['Construction Name', 'EEC', 'ECC']]

    # Import density and thickness for no mass materials
    density_nomass_df = pd.read_csv('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Density_NoMass.csv')

    # Functions needed within the larger function.
    # area of polygon, poly
    def poly_area(poly):
        if len(poly) < 3:  # not a polygon - no area
            return 0
        total = [0, 0, 0]
        N = len(poly)
        for i in range(N):
            vi1 = poly[i]
            vi2 = poly[(i + 1) % N]
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
        return abs(result / 2)

    # unit normal vector of plane defined by points a, b, and c
    def unit_normal(a, b, c):
        x = np.linalg.det([[1, a[1], a[2]],
                           [1, b[1], b[2]],
                           [1, c[1], c[2]]])
        y = np.linalg.det([[a[0], 1, a[2]],
                           [b[0], 1, b[2]],
                           [c[0], 1, c[2]]])
        z = np.linalg.det([[a[0], a[1], 1],
                           [b[0], b[1], 1],
                           [c[0], c[1], 1]])
        magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
        return (x / magnitude, y / magnitude, z / magnitude)

    # Material Extraction to dataframe to csv
    od_material = idf.materials[0]
    material_headers = list(od_material.keys())
    material_df = pd.DataFrame(idf.materials, columns=material_headers)

    # Material No Mass Extraction to dataframe to csv
    od_material_nomass = idf.materialnomasss[0]
    material_nomass_headers = list(od_material_nomass.keys())
    material_nomass_df = pd.DataFrame(idf.materialnomasss, columns=material_nomass_headers)
    material_nomass_df = material_nomass_df.merge(density_nomass_df, how='inner', on='Name')

    # Construction Extraction to dataframe to csv
    od_construction = idf.constructions[0]
    construction_headers = list(od_construction.keys())
    construction_df = pd.DataFrame(idf.constructions, columns=construction_headers)

    # Surface Extraction to DF to csv
    num_cols = []
    for surface in idf.buildingsurfacedetaileds:
        num_cols.append(len(surface))
    index = num_cols.index(max(num_cols))
    od_surface = idf.buildingsurfacedetaileds[index]
    surface_headers = list(od_surface.keys())
    surfaces_detailed_df = pd.DataFrame(idf.buildingsurfacedetaileds, columns=surface_headers)

    # Calculate the area of each surface
    surface_area = []
    for surface in idf.buildingsurfacedetaileds:
        surface_area.append(poly_area(surface.extensibles))
    surfaces_detailed_df['Surface Area'] = surface_area
    surfaces_detailed_df = surfaces_detailed_df[['Name', 'Surface Type', 'Construction Name',
                                                 'Zone Name', 'Surface Area']]

    # Fenestration Extraction to DF to csv
    num_cols = []
    surface_area = []
    for surface in idf.fenestrationsurfacedetaileds:
        # Collect the number of columns for each fenestration surface
        num_cols.append(len(surface))

        # Calculate the area of each fenestration
        vertex_1_x = surface.vertex_1_xcoordinate
        vertex_1_y = surface.vertex_1_ycoordinate
        vertex_1_z = surface.vertex_1_zcoordinate
        vertex_2_x = surface.vertex_2_xcoordinate
        vertex_2_y = surface.vertex_2_ycoordinate
        vertex_2_z = surface.vertex_2_zcoordinate
        vertex_3_x = surface.vertex_3_xcoordinate
        vertex_3_y = surface.vertex_3_ycoordinate
        vertex_3_z = surface.vertex_3_zcoordinate
        vertices = [[vertex_1_x, vertex_1_y, vertex_1_z],
                    [vertex_2_x, vertex_2_y, vertex_2_z],
                    [vertex_3_x, vertex_3_y, vertex_3_z]]
        surface_area.append(poly_area(vertices))
    index = num_cols.index(max(num_cols))
    od_fenestration = idf.fenestrationsurfacedetaileds[index]
    fenestration_headers = list(od_fenestration.keys())
    fenestration_detailed_df = pd.DataFrame(idf.fenestrationsurfacedetaileds, columns=fenestration_headers)
    fenestration_detailed_df['Fenestration Surface Area'] = surface_area
    fenestration_detailed_df = fenestration_detailed_df.merge(LCI_window_df, on='Construction Name')
    fenestration_detailed_df['EE'] = fenestration_detailed_df['EEC'] * fenestration_detailed_df[
        'Fenestration Surface Area']
    fenestration_detailed_df['EC'] = fenestration_detailed_df['ECC'] * fenestration_detailed_df[
        'Fenestration Surface Area']

    fenestration_detailed_df.rename(columns={'Construction Name': 'material'}, inplace=True)

    fenestration_total_EE = sum(fenestration_detailed_df['EE'])
    fenestration_total_EC = sum(fenestration_detailed_df['EC'])

    fenestration_by_mat_EE = fenestration_detailed_df.groupby('material')['EE'].sum()
    fenestration_by_mat_EC = fenestration_detailed_df.groupby('material')['EC'].sum()

    # # Selecting the materials that are used by constructions
    material_list = pd.unique(construction_df[['Outside Layer', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6',
                                               'Layer 7', 'Layer 8', 'Layer 9', 'Layer 10']].values.ravel('K'))
    materials_used_df = material_df.loc[material_df['Name'].isin(material_list)]
    # Adding in EE and EC coefficients from LCI csv file.
    materials_used_df = materials_used_df.merge(LCI_df, on='Name')

    material_nomass_list = pd.unique(construction_df[['Outside Layer', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5',
                                                      'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9',
                                                      'Layer 10']].values.ravel('K'))
    materials_nomass_used_df = material_nomass_df.loc[material_nomass_df['Name'].isin(material_nomass_list)]
    # Adding in EE and EC coefficients from LCI csv file.
    materials_nomass_used_df = materials_nomass_used_df.merge(LCI_df, on='Name')

    # Combining material data with construction data
    construction_df_melted = construction_df.melt(id_vars='Name',
                                                  var_name='Layer',
                                                  value_name='Material')
    material_all_df = pd.concat([materials_used_df, materials_nomass_used_df], axis=0, ignore_index=True, sort=False)
    const_and_mat = pd.merge(construction_df_melted, material_all_df, how='inner', left_on='Material', right_on='Name',
                             suffixes=('_const', '_material')).drop_duplicates().reset_index()

    # Filter out constructions matched with materials to only columns of interest
    const_and_mat = const_and_mat[['Name_const', 'Layer', 'Material', 'Thickness', 'Density', 'EEC', 'ECC']].dropna(
        axis=0, subset=['Material'])
    const_and_mat = const_and_mat.pivot(index='Name_const', columns='Layer').swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Combing construction data with surface data
    surface_and_const = pd.merge(surfaces_detailed_df, const_and_mat, how='inner', left_on='Construction Name',
                                 right_on='Name_const', suffixes=('_surf', 'const'))

    # Creating new column for mass
    num_layers = 0  # counting number of layers
    if max(surface_and_const.columns == ('Outside Layer', 'Density')) == 1:
        surface_and_const['Layer_1_Mass'] = surface_and_const[('Outside Layer', 'Density')] * surface_and_const[
            ('Outside Layer', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 2', 'Density')) == 1:
        surface_and_const['Layer_2_Mass'] = surface_and_const[('Layer 2', 'Density')] * surface_and_const[
            ('Layer 2', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 3', 'Density')) == 1:
        surface_and_const['Layer_3_Mass'] = surface_and_const[('Layer 3', 'Density')] * surface_and_const[
            ('Layer 3', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 4', 'Density')) == 1:
        surface_and_const['Layer_4_Mass'] = surface_and_const[('Layer 4', 'Density')] * surface_and_const[
            ('Layer 4', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 5', 'Density')) == 1:
        surface_and_const['Layer_5_Mass'] = surface_and_const[('Layer 5', 'Density')] * surface_and_const[
            ('Layer 5', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 6', 'Density')) == 1:
        surface_and_const['Layer_6_Mass'] = surface_and_const[('Layer 6', 'Density')] * surface_and_const[
            ('Layer 6', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 7', 'Density')) == 1:
        surface_and_const['Layer_7_Mass'] = surface_and_const[('Layer 7', 'Density')] * surface_and_const[
            ('Layer 7', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 8', 'Density')) == 1:
        surface_and_const['Layer_8_Mass'] = surface_and_const[('Layer 8', 'Density')] * surface_and_const[
            ('Layer 8', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 9', 'Density')) == 1:
        surface_and_const['Layer_9_Mass'] = surface_and_const[('Layer 9', 'Density')] * surface_and_const[
            ('Layer 9', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1
    if max(surface_and_const.columns == ('Layer 10', 'Density')) == 1:
        surface_and_const['Layer_10_Mass'] = surface_and_const[('Layer 10', 'Density')] * surface_and_const[
            ('Layer 10', 'Thickness')] * surface_and_const['Surface Area']
        num_layers = num_layers + 1

    # Creating new column for EE
    if max(surface_and_const.columns == ('Outside Layer', 'Density')) == 1:
        surface_and_const['EE_Layer_1'] = surface_and_const['Layer_1_Mass'] * surface_and_const[
            ('Outside Layer', 'EEC')]
    if max(surface_and_const.columns == ('Layer 2', 'Density')) == 1:
        surface_and_const['EE_Layer_2'] = surface_and_const['Layer_2_Mass'] * surface_and_const[('Layer 2', 'EEC')]
    if max(surface_and_const.columns == ('Layer 3', 'Density')) == 1:
        surface_and_const['EE_Layer_3'] = surface_and_const['Layer_3_Mass'] * surface_and_const[('Layer 3', 'EEC')]
    if max(surface_and_const.columns == ('Layer 4', 'Density')) == 1:
        surface_and_const['EE_Layer_4'] = surface_and_const['Layer_4_Mass'] * surface_and_const[('Layer 4', 'EEC')]
    if max(surface_and_const.columns == ('Layer 5', 'Density')) == 1:
        surface_and_const['EE_Layer_5'] = surface_and_const['Layer_5_Mass'] * surface_and_const[('Layer 5', 'EEC')]
    if max(surface_and_const.columns == ('Layer 6', 'Density')) == 1:
        surface_and_const['EE_Layer_6'] = surface_and_const['Layer_6_Mass'] * surface_and_const[('Layer 6', 'EEC')]
    if max(surface_and_const.columns == ('Layer 7', 'Density')) == 1:
        surface_and_const['EE_Layer_7'] = surface_and_const['Layer_7_Mass'] * surface_and_const[('Layer 7', 'EEC')]
    if max(surface_and_const.columns == ('Layer 8', 'Density')) == 1:
        surface_and_const['EE_Layer_8'] = surface_and_const['Layer_8_Mass'] * surface_and_const[('Layer 8', 'EEC')]
    if max(surface_and_const.columns == ('Layer 9', 'Density')) == 1:
        surface_and_const['EE_Layer_9'] = surface_and_const['Layer_9_Mass'] * surface_and_const[('Layer 9', 'EEC')]
    if max(surface_and_const.columns == ('Layer 10', 'Density')) == 1:
        surface_and_const['EE_Layer_10'] = surface_and_const['Layer_10_Mass'] * surface_and_const[('Layer 10', 'EEC')]

    # Total Embodied Energy
    surface_and_const['Total EE'] = surface_and_const[surface_and_const.columns[-num_layers:]].sum(axis=1)

    # Creating new column for EC
    if max(surface_and_const.columns == ('Outside Layer', 'Density')) == 1:
        surface_and_const['EC_Layer_1'] = surface_and_const['Layer_1_Mass'] * surface_and_const[
            'Outside Layer', 'ECC']
        surface_and_const.rename(columns={('Outside Layer', 'Material'): 'Material_Layer_1'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 2', 'Density')) == 1:
        surface_and_const['EC_Layer_2'] = surface_and_const['Layer_2_Mass'] * surface_and_const[('Layer 2', 'ECC')]
        surface_and_const.rename(columns={('Layer 2', 'Material'): 'Material_Layer_2'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 3', 'Density')) == 1:
        surface_and_const['EC_Layer_3'] = surface_and_const['Layer_3_Mass'] * surface_and_const[('Layer 3', 'ECC')]
        surface_and_const.rename(columns={('Layer 3', 'Material'): 'Material_Layer_3'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 4', 'Density')) == 1:
        surface_and_const['EC_Layer_4'] = surface_and_const['Layer_4_Mass'] * surface_and_const[('Layer 4', 'ECC')]
        surface_and_const.rename(columns={('Layer 4', 'Material'): 'Material_Layer_4'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 5', 'Density')) == 1:
        surface_and_const['EC_Layer_5'] = surface_and_const['Layer_5_Mass'] * surface_and_const[('Layer 5', 'ECC')]
        surface_and_const.rename(columns={('Layer 5', 'Material'): 'Material_Layer_5'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 6', 'Density')) == 1:
        surface_and_const['EC_Layer_6'] = surface_and_const['Layer_6_Mass'] * surface_and_const[('Layer 6', 'ECC')]
        surface_and_const.rename(columns={('Layer 6', 'Material'): 'Material_Layer_6'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 7', 'Density')) == 1:
        surface_and_const['EC_Layer_7'] = surface_and_const['Layer_7_Mass'] * surface_and_const[('Layer 7', 'ECC')]
        surface_and_const.rename(columns={('Layer 7', 'Material'): 'Material_Layer_7'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 8', 'Density')) == 1:
        surface_and_const['EC_Layer_8'] = surface_and_const['Layer_8_Mass'] * surface_and_const[('Layer 8', 'ECC')]
        surface_and_const.rename(columns={('Layer 8', 'Material'): 'Material_Layer_8'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 9', 'Density')) == 1:
        surface_and_const['EC_Layer_9'] = surface_and_const['Layer_9_Mass'] * surface_and_const[('Layer 9', 'ECC')]
        surface_and_const.rename(columns={('Layer 9', 'Material'): 'Material_Layer_9'}, inplace=True)
    if max(surface_and_const.columns == ('Layer 10', 'Density')) == 1:
        surface_and_const['EC_Layer_10'] = surface_and_const['Layer_10_Mass'] * surface_and_const[('Layer 10', 'ECC')]
        surface_and_const.rename(columns={('Layer 10', 'Material'): 'Material_Layer_10'}, inplace=True)

    # Total Embodied Carbon
    surface_and_const['Total EC'] = surface_and_const[surface_and_const.columns[-num_layers:]].sum(axis=1)

    # Calculate total floor area

    floor_area = surfaces_detailed_df[surfaces_detailed_df['Surface Type'] == 'Floor'].sum()
    floor_area = floor_area[4]

    # Detailed summary data frame creation

    # Summarize EE and EC by material
    # List of layers
    col_names_mat = list(surface_and_const)
    mat_names = [name for name in col_names_mat if 'Material' in name]
    mat_names.sort()
    ee_names = [name for name in col_names_mat if ('EE' in name and 'Total' not in name)]
    ee_names.sort()
    mat_ee = mat_names + ee_names


    # List of all materials and EE for each layer
    ee_by_mat_df = surface_and_const[['Name'] + mat_ee]

    ee_by_mat_melt_1 = pd.melt(ee_by_mat_df,
                               id_vars='Name',
                               value_vars=mat_names,
                               var_name='material_by_layer',
                               value_name='material')
    ee_by_mat_melt_2 = pd.melt(ee_by_mat_df,
                               id_vars='Name',
                               value_vars=ee_names,
                               var_name='EE_by_layer',
                               value_name='EE')

    ee_by_mat_melt_1.index = range(0, len(ee_by_mat_melt_1))
    ee_by_mat_melt_2.index = range(0, len(ee_by_mat_melt_2))
    ee_by_mat_melt = (pd.concat([ee_by_mat_melt_1, ee_by_mat_melt_2], axis=1))
    ee_by_mat_summary = ee_by_mat_melt.groupby('material')['EE'].sum()

    # List of all materials and EC for each layer
    ec_names = [name for name in col_names_mat if ('EC' in name and 'Total' not in name)]
    ec_names.sort()
    mat_ec = mat_names + ec_names
    mat_ec.sort()
    ec_by_mat_df = surface_and_const[['Name'] + mat_ec]

    ec_by_mat_melt_1 = pd.melt(ec_by_mat_df,
                               id_vars='Name',
                               value_vars=mat_names,
                               var_name='material_by_layer',
                               value_name='material')
    ec_by_mat_melt_2 = pd.melt(ec_by_mat_df,
                               id_vars='Name',
                               value_vars=ec_names,
                               var_name='EC_by_layer',
                               value_name='EC')
    ec_by_mat_melt_1.index = range(0, len(ec_by_mat_melt_1))
    ec_by_mat_melt_2.index = range(0, len(ec_by_mat_melt_2))
    ec_by_mat_melt = (pd.concat([ec_by_mat_melt_1, ec_by_mat_melt_2], axis=1))
    ec_by_mat_summary = ec_by_mat_melt.groupby('material')['EC'].sum()

    summary_bymaterial = pd.concat([ee_by_mat_summary, ec_by_mat_summary], axis=1)
    summary_bymaterial.reset_index(inplace=True)
    summary_bymaterial = summary_bymaterial.append({'material': fenestration_by_mat_EE.index.values[0],
                                                    'EE': fenestration_by_mat_EE[0],
                                                    'EC': fenestration_by_mat_EC[0]},
                                                   ignore_index=True)
    summary_bymaterial['EE_normalized'] = summary_bymaterial['EE']/floor_area
    summary_bymaterial['EC_normalized'] = summary_bymaterial['EC']/floor_area
    summary_bymaterial['Building'] = building_name

    # Summary By surface type:
    ee_by_surf_type = surface_and_const.groupby('Surface Type')['Total EE'].sum()
    ec_by_surf_type = surface_and_const.groupby('Surface Type')['Total EC'].sum()

    summary_bysurface = pd.concat([ee_by_surf_type, ec_by_surf_type], axis=1)
    summary_bysurface.reset_index(inplace=True)
    summary_bysurface = summary_bysurface.append({'Surface Type': 'Windows',
                                                    'Total EE': fenestration_by_mat_EE[0],
                                                    'Total EC': fenestration_by_mat_EC[0]},
                                                 ignore_index=True)
    summary_bysurface['EE_Normalized'] = summary_bysurface['Total EE'] / floor_area
    summary_bysurface['EC_Normalized'] = summary_bysurface['Total EC'] / floor_area
    summary_bysurface['Building'] = building_name

    # Create a new sheet that summarizes the total embodied energy and embodied carbon.
    summary_dict = {'Total EE [MJ]': [sum(surface_and_const['Total EE'], fenestration_total_EE)],
                    'Total EC [kg CO2e]': [sum(surface_and_const['Total EC'], fenestration_total_EC)],
                    'Total Floor Area': [floor_area],
                    'Normalized EE [MJ/m2]': [sum(surface_and_const['Total EE'], fenestration_total_EE) / floor_area],
                    'Normalized EC [kg CO2e/m2]': [
                        sum(surface_and_const['Total EC'], fenestration_total_EC) / floor_area],
                    'Building': [building_name]}
    summary = pd.DataFrame(data=summary_dict)

    # Writing files to Excel
    with pd.ExcelWriter(file_name) as writer:
        summary_bysurface.to_excel(writer, sheet_name="summary_by_surface")
        summary_bymaterial.to_excel(writer, sheet_name="summary_by_material")
        summary.to_excel(writer, sheet_name="summary")
        surface_and_const.to_excel(writer, sheet_name='surface_materials')
        material_df.to_excel(writer, sheet_name="materials")
        material_nomass_df.to_excel(writer, sheet_name="materialsnomass")
        construction_df.to_excel(writer, sheet_name="constructions")
        surfaces_detailed_df.to_excel(writer, sheet_name="surfaces_detailed")
        fenestration_detailed_df.to_excel(writer, sheet_name="fenestrationsdetailed")

    print(file_name, ' has been written! **')
    print(summary.head())

    return summary, summary_bysurface, summary_bymaterial



# Set directory for analysis
# os.chdir('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ2A_Tampa_90.1_2016')
# file_name = '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Results Summary/CZ2A_Tampa_summary.xlsx'

# os.chdir('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ4C_Seattle_90.1_2016')
# file_name = '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Results Summary/CZ4C_Seattle_summary.xlsx'
#
os.chdir('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ5B_Denver_90.1_2016')
file_name = '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Results Summary/CZ5B_Denver_summary.xlsx'
#
# os.chdir('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ7_InternationalFalls_90.1_2016')
# file_name = '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Results Summary/CZ7_InternationalFalls_summary.xlsx'

mypath = os.getcwd()

file_list = [f for f in listdir(mypath) if f.endswith('idf')]
print(file_list)

summary_df = pd.DataFrame()
summary_bysurface_df = pd.DataFrame()
summary_bymaterial_df = pd.DataFrame()
for file in file_list:
    new_df_1, new_df_2, new_df_3 = idf_materials_toexcel(file)
    # print(new_df_1, new_df_2, new_df_3)
    summary_df = summary_df.append(new_df_1)
    summary_bysurface_df = summary_bysurface_df.append(new_df_2)
    summary_bymaterial_df = summary_bymaterial_df.append(new_df_3)
#

# Just do one file to excel
# summary_df, summary_bysurface_df, summary_bymaterial_df  = idf_materials_toexcel(
#     '/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/CZ5B_Denver_90.1_2016/ASHRAE90.1_ApartmentHighRise_STD2016_Denver.idf')

# summary_df.to_excel('/Users/josepharehart/PycharmProjects/DOE_Ref_Materials/Results Summary/CZ5B_Denver_summary.xlsx')


with pd.ExcelWriter(file_name) as writer:
    summary_df.to_excel(writer, sheet_name="summary")
    summary_bysurface_df.to_excel(writer, sheet_name="summary_by_surface")
    summary_bymaterial_df.to_excel(writer, sheet_name="summary_by_material")
