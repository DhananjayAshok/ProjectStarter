from utils.parameter_handling import load_parameters
from tqdm import tqdm
import numpy as np

def paired_bootstrap(sys1, sys2, num_samples=10000, sample_ratio=0.5, progress_title=None, parameters=None):
  ''' Evaluate with paired boostrap

  This compares two systems, performing a significance tests with
  paired bootstrap resampling to compare the performance of the two systems.
  
  :param sys1: The eval metrics (instance-wise) of system 1
  :param sys2: The eval metrics (instance-wise) of system 2. Must be of the same length
  :param num_samples: The number of bootstrap samples to take
  :param sample_ratio: The ratio of samples to take every time
  '''
  parameters = load_parameters(parameters)

  sys1_scores = []
  sys2_scores = []
  wins = [0, 0, 0]
  n = len(sys1)
  if 
  ids = list(range(n))

  for _ in tqdm(range(num_samples), desc=progress_title):
    # Subsample the gold and system outputs
    reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio),replace=True)
    reduced_sys1 = [sys1[i] for i in reduced_ids]
    reduced_sys2 = [sys2[i] for i in reduced_ids]
    # Calculate accuracy on the reduced sample and save stats
    sys1_score = np.mean(reduced_sys1)
    sys2_score = np.mean(reduced_sys2)
    if sys1_score > sys2_score:
      wins[0] += 1
    elif sys1_score < sys2_score:
      wins[1] += 1
    else:
      wins[2] += 1
    sys1_scores.append(sys1_score)
    sys2_scores.append(sys2_score)

  # Print win stats
  wins = [x/float(num_samples) for x in wins]
  print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
  if wins[0] > wins[1]:
    print('(sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
  elif wins[1] > wins[0]:
    print('(sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))

  # Print system stats
  sys1_scores.sort()
  sys2_scores.sort()
  print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
  print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
