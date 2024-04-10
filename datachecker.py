import env_manager as em

for i in range(250, 585):
    path = f'./envData/env{i}/data_points.dat'
    em.load_data_points(path)