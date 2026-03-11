from typing import Any, Optional
from collections.abc import Sequence
from utils.parameter_handling import load_parameters
from utils.log_handling import log_warn, log_info
from tqdm import tqdm
import numpy as np

def paired_bootstrap(sys1: Sequence[float], sys2: Sequence[float], num_samples: int = 10000, sample_ratio: float = 0.5, progress_title: Optional[str] = None, parameters: Optional[dict[str, Any]] = None) -> None:
  ''' Evaluate with paired bootstrap.

  This compares two systems, performing a significance test with
  paired bootstrap resampling to compare the performance of the two systems.

  :param sys1: The eval metrics (instance-wise) of system 1.
  :type sys1: Sequence[float]
  :param sys2: The eval metrics (instance-wise) of system 2. Must be of the same length.
  :type sys2: Sequence[float]
  :param num_samples: The number of bootstrap samples to take.
  :type num_samples: int
  :param sample_ratio: The ratio of instances to sample on each bootstrap iteration.
  :type sample_ratio: float
  :param progress_title: Optional label shown on the tqdm progress bar and in logged results.
  :type progress_title: str or None
  :param parameters: Loaded parameters dict for logging. If None, logs to console only.
  :type parameters: dict[str, Any] or None
  :return: None. Results are logged via log_info.
  :rtype: None
  '''
  parameters = load_parameters(parameters)

  sys1_scores = []
  sys2_scores = []
  wins = [0, 0, 0]
  n = len(sys1)
  if len(sys2) != n:
    log_warn("System outputs must be of the same length for paired bootstrap evaluation." , parameters)
    return
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

  # log_info win stats
  if progress_title:
    log_info(f"Results: {progress_title}", parameters=parameters)
  wins = [x/float(num_samples) for x in wins]
  log_info('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]), parameters=parameters)
  if wins[0] > wins[1]:
    log_info('(sys1 is superior with p value p=%.3f)\n' % (1-wins[0]), parameters=parameters)
  elif wins[1] > wins[0]:
    log_info('(sys2 is superior with p value p=%.3f)\n' % (1-wins[1]), parameters=parameters)

  # log_info system stats
  sys1_scores.sort()
  sys2_scores.sort()
  log_info('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]), parameters=parameters)
  log_info('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]), parameters=parameters)
