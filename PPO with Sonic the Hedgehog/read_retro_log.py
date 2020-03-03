# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 07:05:14 2020

@author: pc
"""

from pandas import read_csv, concat
from matplotlib import pyplot as plt

all_logs = [# 'openai-2020-02-29-09-58-01-954747',
            # 'openai-2020-02-29-10-21-19-856632',
            # 'openai-2020-02-29-10-28-26-224523',
            'openai-2020-02-29-11-05-20-496981',
            'openai-2020-02-29-11-11-05-903608',
            'openai-2020-02-29-12-54-48-999003',
            'openai-2020-02-29-15-59-36-489930',
            'openai-2020-02-29-19-38-43-050516',
            'openai-2020-03-01-08-20-05-149529',
            'openai-2020-03-01-14-56-59-852856',
            'openai-2020-03-01-21-56-15-568347'
]

df_l = read_csv(r'C:\Users\pc\AppData\Local\Temp\openai-2020-02-29-07-16-42-989292\progress.csv')

for sl in all_logs:
    log_path = 'C:\\Users\\pc\\AppData\\Local\\Temp\\' + sl + '\\progress.csv'
    tl = read_csv(log_path)
    df_l = concat([df_l, tl])

df_l.shape

df_l['Mean score test level'].max()

df_l['Mean score test level'].plot()

# df_l is a pandas dataframe with a that was score was generated 
# every 10 updates.

np_mstl = df_l['Mean score test level'].values
x_labels = [str(u) for u in range(0, len(np_mstl)*10, 10)]

plt.plot(np_mstl, linewidth=0.75)
plt.title('Mean score on test level during training.')
plt.gca().set_xlabel('Updates')
plt.gca().set_ylabel('Mean score')
plt.xticks(range(0, 435, 40), x_labels[::40])
plt.hlines(33.6, 0, 435, color='red', linestyles='dashed', linewidth=0.75)
#plt.gca().set_xticklabels(x_labels)
plt.show()
