import pandas as pd
import joblib

def clean_trees_data(file, 
                     percent=50, 
                     print_columns_dropped=False, 
                     predict=False):

    # Load data
    df = pd.read_csv(f'../data/raw/{file}.csv', sep=',', header = 0, index_col=False,names=None)

    # Removing empty rows and columns
    df = df.dropna(how='all').dropna(how='all', axis=1)

    # Remove duplicated or useless columns
    columns_to_drop = ['elem_point_id', 'code', 'nom', 'genre', 'genre_desc', 'categorie', 'categorie_desc', 'sous_categorie_desc', 'code_parent_desc', 'bien_reference']
    if print_columns_dropped:
        print("* Useless and Duplicated Columns dropped:", columns_to_drop, sep='\n'*2)
    df = df.drop(columns=columns_to_drop)

    # Delete columns which have missing values above a given percent
    percent_of_na = pd.DataFrame(df.isna().sum())
    percent_of_na.columns = ['percent']
    percent_of_na.percent = percent_of_na.percent*100/len(df)
    columns_with_missing = percent_of_na[percent_of_na['percent']>percent].index.tolist()
    if ("anneedeplantation" in columns_with_missing):
        columns_with_missing.remove("anneedeplantation")
    if print_columns_dropped:
        print(f"\n* Columns with a minimum of {percent}% missing values dropped:\n",
              percent_of_na[percent_of_na['percent']>percent])
    df = df.drop(columns=columns_with_missing)

    # Get geolocation clusters
    df[['latitude', 'longitude']] = df['geo_point_2d'].str.split(",", expand = True).apply(pd.to_numeric)
    loaded_model = joblib.load('../src/gamt/geolocation_clustering.joblib')
    df['geo_cluster'] = loaded_model.predict(df[['latitude','longitude']])
    df = df.drop(columns='geo_point_2d')

    # Manage string columns with multiple items
    code_parent = ['ESP1200', 'ESP517', 'ESP670', 'ESP67', 'ESP1214', 'ESP1059', 'ESP1219', 'ESP181', 'ESP623', 'ESP1290', 'ESP38', 'ESP460', 'ESP1120', 'ESP1029', 'ESP668', 'ESP41509', 'ESP630', 'ESP399', 'ESP1204', 'ESP750', 'ESP336', 'ESP402', 'ESP254', 'ESP988', 'ESP1152', 'ESP766', 'ESP567', 'ESP361', 'ESP589', 'ESP1198', 'ESP1291', 'ESP895', 'ESP565', 'ESP1069', 'ESP940', 'ESP1363', 'ESP1365', 'ESP104', 'ESP1348', 'ESP33631', 'ESP1427', 'ESP841', 'ESP1255', 'ESP883', 'ESP238', 'ESP82', 'ESP767', 'ESP327', 'ESP158', 'ESP493', 'ESP997', 'ESP682', 'ESP31362', 'ESP1252', 'ESP40', 'ESP31418', 'ESP828', 'ESP235', 'ESP31431', 'ESP1355', 'ESP461', 'ESP95', 'ESP227', 'ESP896', 'ESP812', 'ESP263', 'ESP853', 'ESP645', 'ESP1293', 'ESP413', 'ESP1285', 'ESP679', 'ESP10', 'ESP230', 'ESP555', 'ESP1295', 'ESP41930', 'ESP1279', 'ESP267', 'ESP975', 'ESP1137', 'ESP1323', 'ESP664', 'ESP41935', 'ESP500', 'ESP17', 'ESP1010', 'ESP21', 'ESP1320', 'ESP968', 'ESP78', 'ESP1206', 'ESP129', 'ESP911', 'ESP1009', 'ESP1329', 'ESP366', 'ESP1395', 'ESP315', 'ESP631', 'ESP554', 'ESP1032', 'ESP125', 'ESP13', 'ESP354', 'ESP132', 'ESP929', 'ESP1133', 'ESP935', 'ESP866', 'ESP582', 'ESP754', 'ESP642', 'ESP798', 'ESP521', 'ESP454', 'ESP185', 'ESP1318', 'ESP931', 'ESP240', 'ESP1079', 'ESP1350', 'ESP786', 'ESP1194', 'ESP396', 'ESP1215', 'ESP171', 'ESP331', 'ESP1046', 'ESP657', 'ESP1026', 'ESP64', 'ESP347', 'ESP339', 'ESP292', 'ESP1368', 'ESP1426', 'ESP790', 'ESP202', 'ESP1246', 'ESP490', 'ESP276', 'ESP579', 'ESP43', 'ESP31408', 'ESP1210', 'ESP862', 'ESP1169', 'ESP259', 'ESP644', 'ESP69', 'ESP821', 'ESP770', 'ESP298', 'ESP309', 'ESP1239', 'ESP949', 'ESP744', 'ESP1007', 'ESP851', 'ESP200', 'ESP1062', 'ESP41926', 'ESP818', 'ESP1115', 'ESP1040', 'ESP785', 'ESP1419', 'ESP1420', 'ESP824', 'ESP923', 'ESP1425', 'ESP1266', 'ESP1135', 'ESP1254', 'ESP813', 'ESP166', 'ESP1216', 'ESP272', 'ESP1188', 'ESP228', 'ESP32', 'ESP100', 'ESP1191', 'ESP728', 'ESP282', 'ESP314', 'ESP419', 'ESP89', 'ESP1003', 'ESP295', 'ESP408', 'ESP699', 'ESP800', 'ESP939', 'ESP1373', 'ESP1065', 'ESP1172', 'ESP1319', 'ESP123', 'ESP924', 'ESP41515', 'ESP1415', 'ESP1199', 'ESP1123', 'ESP1145', 'ESP938', 'ESP36656', 'ESP739', 'ESP38626', 'ESP513', 'ESP1376', 'ESP967', 'ESP31', 'ESP473', 'ESP512', 'ESP1362', 'ESP1205', 'ESP892', 'ESP1154', 'ESP401', 'ESP583', 'ESP363', 'ESP73', 'ESP598', 'ESP1138', 'ESP605', 'ESP475', 'ESP742', 'ESP507', 'ESP1070', 'ESP1182', 'ESP321', 'ESP92', 'ESP1192', 'ESP15', 'ESP1286', 'ESP1099', 'ESP1316', 'ESP68', 'ESP287', 'ESP1236', 'ESP596', 'ESP716', 'ESP477', 'ESP288', 'ESP533', 'ESP550', 'ESP113', 'ESP926', 'ESP164', 'ESP128', 'ESP1289', 'ESP1325', 'ESP343', 'ESP806', 'ESP1189', 'ESP1225', 'ESP138', 'ESP61', 'ESP41949', 'ESP14', 'ESP41947', 'ESP1202', 'ESP796', 'ESP1248', 'ESP1072', 'ESP242', 'ESP1380', 'ESP199', 'ESP1176', 'ESP264', 'ESP1181', 'ESP590', 'ESP986', 'ESP1105', 'ESP1315', 'ESP803', 'ESP1082', 'ESP731', 'ESP71', 'ESP239', 'ESP526', 'ESP780', 'ESP568', 'ESP41', 'ESP349', 'ESP650', 'ESP617', 'ESP412', 'ESP1028', 'ESP1066', 'ESP799', 'ESP168']
    genre_bota = ['Idesia', 'Larix', 'Clerodendron', 'Ptelea', 'Punica', 'Pterostyrax', 'Styrax', 'Hibiscus', 'Eriobotrya ', 'Lonicera', 'Cephalotaxus', 'Chimonanthus', 'Pseudotsuga', 'Phillyrea', 'Cryptomeria', 'Araucaria', 'Hovenia', 'Castanea', 'Acacia', 'Rhamnus', 'Staphylea', 'Nyssa', 'Poncinos', 'Sorbopyrus', 'Eucalyptus', 'Sterculia']
    espece = ['pyrifolia', 'nootkatensis', 'florida', 'tartaricum', 'simonii', 'decidua', 'tricotonum', 'purpurea', 'virginicus', 'polycarpa', 'spectabilis', 'sinense', 'hispida', 'hybride', 'sepulcralis', 'bicolor', 'sieboldiana', 'aucuparia', 'omorika', 'euchlora', 'triflorum', 'granatum', 'trifoliata', 'trojana', 'caprea', 'concolor', 'pinea', 'pavia', 'harringtonia', 'alnifolia', 'parviflora', 'carpinifolium', 'syriacus', 'racemosa', 'thuringiaca', 'cuspidata', 'hupehensis', 'everest', 'sanguinea', 'campanulata', 'turneri', 'galaxy', 'araucana', 'prunifolia', 'menziesii', 'bungeana', 'amygdalus', 'elaeagnifolia', 'stellata', 'cordiformis', 'quadrangulata', 'praecox', 'palmatum', 'pinsapo', 'rosmarinifolia', 'laciniosa', 'heldreichii', 'serotina', 'macrophyllum', 'pisifera', 'auricularis', 'crenata', 'dealbata', 'macrolepis', 'foetida', 'koreana', 'mahaleb', 'gunii', 'heptapeta', 'sativa', 'drupacea', 'monticola', 'formosensis', 'coccinea', 'macrocarpa', 'zoeschense', 'griffithii', 'kentukea', 'ioenis', 'denboerii', 'cachemeriana', 'liliflora', 'mugo', 'bumalda', 'alaternus', 'chamaemespilus', 'opalus', 'strobus']
    features_with_others = ['code_parent', 'genre_bota', 'espece']
    for col in features_with_others:
        for row in locals()[col]:
            df[col] = df[col].replace(row,'Others')
    
    # Keep predict/model data
    if predict:
        df = df.loc[df['anneedeplantation'].isna()]
    else:
        df = df.dropna(subset=['anneedeplantation'])

    return df