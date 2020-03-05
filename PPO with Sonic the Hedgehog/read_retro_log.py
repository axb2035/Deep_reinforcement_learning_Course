# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 07:05:14 2020

@author: pc
"""

from pandas import read_csv, concat
from matplotlib import pyplot as plt

all_logs = [
            'openai-2020-03-04-21-03-16-532842',
            'openai-2020-03-05-18-21-57-270804',
            'openai-2020-03-06-06-17-50-001558'
            ]

df_l = read_csv(r'C:\Users\pc\AppData\Local\Temp\openai-2020-03-04-17-40-42-927222\progress.csv')

for sl in all_logs:
    log_path = 'C:\\Users\\pc\\AppData\\Local\\Temp\\' + sl + '\\progress.csv'
    tl = read_csv(log_path)
    df_l = concat([df_l, tl])

df_l.shape

df_l.rename(columns={"Mean score - test (1 level)" : "test score"}, inplace=True)

df_l['test score'].max()

df_l['test score'].plot()

# df_l is a pandas dataframe with a that was score was generated 
# every 10 updates.

np_mstl = df_l['test score'].values
np_train_loss = df_l['policy_loss'].values
np_value_loss = df_l['value_loss'].values
x_len = len(np_mstl)*10
x_labels = [str(u) for u in range(0, x_len, 10)]

plt.plot(np_mstl, linewidth=0.75)
plt.title('Mean score on test level during training.')
plt.gca().set_xlabel('Updates')
plt.gca().set_ylabel('Mean score')
plt.xticks(range(0, 150, 15), x_labels[::15])
# plt.hlines(33.6, 0, 57, color='red', linestyles='dashed', linewidth=0.75)
#plt.gca().set_xticklabels(x_labels)
plt.show()

# Plot the train and value loss.

plt.plot(np_train_loss, label='Train', linewidth=0.75)
plt.title('Train / value loss during training.')
plt.gca().set_xlabel('Updates')
plt.gca().set_ylabel('Train loss')
plt.xticks(range(0, x_len, 16), x_labels[::16])
plt.gca().legend(loc=2)
# plt.hlines(33.6, 0, 57, color='red', linestyles='dashed', linewidth=0.75)
#plt.gca().set_xticklabels(x_labels)

ax2 = plt.gca().twinx() 
plt.plot(np_value_loss, label='Value', linewidth=0.75, color='orange')
plt.gca().set_ylabel('Value loss')
ax2.legend(loc=0)
plt.show()
