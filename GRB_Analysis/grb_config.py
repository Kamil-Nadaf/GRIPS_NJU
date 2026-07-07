import pandas as pd

grbs_df = pd.DataFrame([
    {'name': 'GRB131011A', 'ra': 32.526, 'dec': -4.411,   'utc': '2013-10-11T17:47:34.99'},
    # {'name': 'GRB140606B', 'ra': 328.12501, 'dec': 32.01458,   'utc': '2014-06-06T03:11:51.86'},
    # {'name': 'GRB150514A', 'ra': 74.8750,  'dec': -60.9691, 'utc': '2015-05-14T18:35:05.35'},
    # {'name': 'GRB151027A', 'ra': 272.48695, 'dec': +61.35344,   'utc': '2015-10-27T03:58:24'},
    # {'name': 'GRB190829A', 'ra': 44.54402, 'dec': -8.95837,   'utc': '2019-08-29T19:56:44.60'},
])

grbs_df = grbs_df.sort_values('name').reset_index(drop=True)

grbs_df['sel_dets'] = [
    ['b1','n9','na','nb'],
    # ['b0','n3','n4','n8'],
    # ['b0','n3','n6','n7'],
    # ['b0','n0','n1','n3'],
    # ['n6','n7','n9'],
]
