import pandas
df = pandas.read_csv("R1/d3.csv", sep=";")
df = df[df['product'] == 'STARFRUIT']
df = df["mid_price"]

# convert to list
data = df.values.tolist()

import numpy as np

coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
intercept = 4.481696494462085

def evaluate(a, b, c, d, intercept):
  coef = [a, b, c, d]
  back = []
  last4 = []
  for i in range(0, len(data)):
    if i < 4:
      last4.append(data[i])
      back.append(None)
    else:
      price = intercept
      for j in range(4):
        price += coef[j] * last4[j]
      back.append(price)
      last4.pop(0)
      last4.append(data[i])

  sumdiff = sum([abs(data[i] - back[i]) for i in range(4, len(data))])
  return abs(sumdiff)
  
back = []
last4 = []
for i in range(0, len(data)):
  if i < 4:
    last4.append(data[i])
    back.append(None)
  else:
    price = intercept
    for j in range(4):
      price += coef[j] * last4[j]
    back.append(price)
    last4.pop(0)
    last4.append(data[i])
    
df2 = pandas.DataFrame(back)

# graph df and df2 on the same plot
df = pandas.DataFrame(data)
df = df.rename(columns={0: "mid_price"})
df2 = df2.rename(columns={0: "back"})
df = df.join(df2)
print(df)
df.plot()
print(np.corrcoef(df["mid_price"][4:], df["back"][4:])[0, 1])



import skopt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

space  = [Real(-1, 1, name='a'),
          Real(-1, 1, name='b'),
          Real(-1, 1, name='c'),
          Real(-1, 1, name='d'), 
          Real(-10, 10, name='intercept')]
@use_named_args(space)
def objective(a, b, c, d, intercept):
    return evaluate(a, b, c, d, intercept)


# res = gp_minimize(objective, space, verbose=True, x0 = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892, 4.481696494462085])

# # Increase the number of calls
# res = gp_minimize(objective, space, n_calls=200, verbose=True, x0 = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892, 4.481696494462085])

# print(res.x)
# print(1-res.fun)

# # Use a different initial point
# res = gp_minimize(objective, space, n_calls=100, verbose=True, x0 = [0, 0, 0, 0, 0])

# print(res.x)
# print(1-res.fun)

# # Use a different optimizer (e.g., forest_minimize for Random Forest-based optimization)
# from skopt import forest_minimize
# res = forest_minimize(objective, space, n_calls=100, verbose=True, x0 = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892, 4.481696494462085])

# print(res.x)
# print(1-res.fun)



