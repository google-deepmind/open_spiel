import numpy as np
import copy
#
# def lagging_mean(li,lag=3):
#   """
#   Calcualte the lagging mean of list given
#   Params:
#     li : the one dimensional list to be processed
#     lag: length of moving average
#   """
#   if len(li) <= lag:
#     return list(np.cumsum(li)/(np.arange(len(li))+1))
#   else:
#     first_half = np.cumsum(li[0:lag-1])/(np.arange(lag-1)+1)
#     ret = np.cumsum(li)
#     ret[lag:] = ret[lag:] - ret[:-lag]
#     second_half = ret[lag - 1:] / lag
#     return first_half.tolist() + second_half.tolist(), 1
#
# a = [1,2,3,4,5,6,7]
# p = lagging_mean(a)
# print(p)

param_dict = {'ars_learning_rate': list(np.round(np.arange(0.01, 0.1, 0.01), decimals=2)),
                  'noise': list(np.arange(0.01, 0.1, 0.01))}
print(param_dict)