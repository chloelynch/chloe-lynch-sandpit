# -*- coding: utf-8 -*-
# Stone Soup 04 - Data Association

# Some general imports and set up
import matplotlib
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import timedelta
from datetime import datetime

import numpy as np

# Simulate Data 
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

# Figure to plot truth (and future data)
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

truth = GroundTruthPath()
start_time = datetime.now()
for n in range(1, 21):
    x = n
    y = n
    varxy = np.array([[0.05,0],[0,0.05]])
    xy = np.random.multivariate_normal(np.array([x,y]),varxy)
    truth.append(GroundTruthState(np.array([[xy[0]], [xy[1]]]), timestamp=start_time+timedelta(seconds=n)))

#Plot the result
ax.plot([state.state_vector[0, 0] for state in truth], 
        [state.state_vector[1, 0] for state in truth], 
        color='g', linestyle="--")

from scipy.stats import multivariate_normal

from stonesoup.types.detection import Detection

measurementss = [set() for _ in range(20)]
for n, state in enumerate(truth):
    if np.random.rand() <= 0.85: # Probability of detection
        x, y = multivariate_normal.rvs(
            state.state_vector.ravel(), cov=np.diag([0.25, 0.25]))
        measurementss[n].add(Detection(
            np.array([[x], [y]]), timestamp=state.timestamp))
    
# Plot the result
ax.scatter([state.state_vector[0, 0] for measurements in measurementss for state in measurements], 
           [state.state_vector[1, 0] for measurements in measurementss for state in measurements], 
           color='b')
fig

from scipy.stats import uniform

from stonesoup.types.detection import Clutter

clutter = []
for n in range(1, 21):
    clutter.append(set())
    for _ in range(np.random.randint(10)):
        x = uniform.rvs(0, 20)
        y = uniform.rvs(0, 20)
        clutter[-1].add(Clutter(
            np.array([[x], [y]]), timestamp=start_time+timedelta(seconds=n)))
# Plot the result
ax.scatter([state.state_vector[0, 0] for clutter_set in clutter for state in clutter_set], 
           [state.state_vector[1, 0] for clutter_set in clutter for state in clutter_set], 
           color='y', marker='2')
fig

# Create Models and Kalman Filter

from scipy.stats import uniform

from stonesoup.types.detection import Clutter

clutter = []
for n in range(1, 21):
    clutter.append(set())
    for _ in range(np.random.randint(10)):
        x = uniform.rvs(0, 20)
        y = uniform.rvs(0, 20)
        clutter[-1].add(Clutter(
            np.array([[x], [y]]), timestamp=start_time+timedelta(seconds=n)))
# Plot the result
ax.scatter([state.state_vector[0, 0] for clutter_set in clutter for state in clutter_set], 
           [state.state_vector[1, 0] for clutter_set in clutter for state in clutter_set], 
           color='y', marker='2')
fig
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.05), ConstantVelocity(0.05)))

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    4, # Number of state dimensions (position and velocity in 2D)
    (0,2), # Mapping measurement vector index to state index
    np.array([[0.25, 0],  # Covariance matrix for Gaussian PDF
              [0, 0.25]])
    )
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([0.25, 0.1, 0.25, 0.1]), timestamp=start_time)

from stonesoup.types.track import Track

track = Track([prior])
for n, (measurements, clutter_set) in enumerate(zip(measurementss, clutter), 1):
    detections = clutter_set.copy()
    detections.update(measurements) # Add measurements and clutter together
    
    hypotheses = data_associator.associate({track}, detections, start_time+timedelta(seconds=n))
    hypothesis = hypotheses[track]
    
    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else: # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

# Plot the resulting track
ax.plot([state.state_vector[0, 0] for state in track[1:]],  # Skip plotting the prior
        [state.state_vector[2, 0] for state in track[1:]],
        marker=".")
fig

from matplotlib.patches import Ellipse
for state in track[1:]: # Skip the prior
    w, v = np.linalg.eig(measurement_model.matrix()@state.covar@measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0,2), 0],
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)
fig

