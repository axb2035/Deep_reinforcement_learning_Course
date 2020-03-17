# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 07:05:14 2020

@author: pc
"""

from pandas import read_csv, concat, DataFrame
from matplotlib import pyplot as plt

# Load the checkpoint performance logs
# Temp logs sored here: C:\Users\pc\AppData\Local\Temp\
df_l = read_csv(r'.\logs\openai-2020-03-06-08-09-25-729383\progress.csv')
df_l = DataFrame()

all_logs = [
            # Additional log directories here.
            ['openai-2020-03-08-12-00-38-624350', 1, 37],
            ['openai-2020-03-09-16-40-41-735069', 0, 0],
            ['openai-2020-03-10-15-59-21-777278', 0, 0],
            ['openai-2020-03-11-22-49-40-427426', 0, 0],
            ['openai-2020-03-12-12-15-29-400823', 0, 0],
            ['openai-2020-03-13-20-26-41-668112', 0, 0],
            ['openai-2020-03-14-22-55-58-698089', 0, 0],
            ['openai-2020-03-17-13-02-42-941418', 0, 0]
            ]

# Append additional logs.
for sl in all_logs:
    log_path = '.\\logs\\' + sl[0] + '\\progress.csv'
    tl = read_csv(log_path)
    if (sl[1] > 0) or (sl[2] > 0) :
        tl = tl[sl[1]:sl[2]]
    df_l = concat([df_l, tl])

df_l.shape

df_l.rename(columns={"Mean score - test (1 level)" : "test score"}, inplace=True)

df_l['test score'].max()/0.01

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
plt.xticks(range(0, x_len//10, x_len//100), x_labels[::x_len//100])
# plt.hlines(33.6, 0, 57, color='red', linestyles='dashed', linewidth=0.75)
#plt.gca().set_xticklabels(x_labels)
plt.show()

# Plot the train and value loss.

plt.plot(np_train_loss, label='Train', linewidth=0.75)
plt.title('Train / value loss during training.')
plt.gca().set_xlabel('Updates')
plt.gca().set_ylabel('Train loss')
plt.xticks(range(0, x_len, x_len//100), x_labels[::x_len//100])
plt.gca().legend(loc=2)
# plt.hlines(33.6, 0, 57, color='red', linestyles='dashed', linewidth=0.75)
#plt.gca().set_xticklabels(x_labels)

ax2 = plt.gca().twinx() 
plt.plot(np_value_loss, label='Value', linewidth=0.75, color='orange')
plt.gca().set_ylabel('Value loss')
ax2.legend(loc=0)
plt.show()
