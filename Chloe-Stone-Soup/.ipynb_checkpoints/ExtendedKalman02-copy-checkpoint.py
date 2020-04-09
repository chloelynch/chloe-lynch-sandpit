# -*- coding: utf-8 -*-
# Stone Soup 02 - Extended Kalman
# Some general imports and set up

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
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
    varxy = np.array([[0.1,0],[0,0.1]])
    xy = np.random.multivariate_normal(np.array([x,y]),varxy)
    truth.append(GroundTruthState(np.array([[xy[0]], [xy[1]]]), timestamp=start_time+timedelta(seconds=n)))
    
#Plot the result
ax.plot([state.state_vector[0, 0] for state in truth], 
        [state.state_vector[1, 0] for state in truth], 
        linestyle="--")

from scipy.stats import multivariate_normal

from stonesoup.types.detection import Detection
from stonesoup.types.angle import Bearing
from stonesoup.functions import cart2pol, pol2cart

measurements = []
sensor_x = 10
sensor_y = 0
for state in truth:
    x = state.state_vector[0, 0]
    y = state.state_vector[1, 0]
    delta_x = (x - sensor_x)
    delta_y = (y - sensor_y)
    #rho, phi = multivariate_normal.rvs(
    rho, phi = np.random.multivariate_normal(
        cart2pol(delta_x, delta_y),
        np.diag([1, np.radians(0.2)]))
    # Special Bearing type used to allow difference in angle calculations    
    measurements.append(Detection(
        np.array([[Bearing(phi)], [rho]]), timestamp=state.timestamp))
    
# Plot the result (back in cart.)
x, y = pol2cart(
    np.hstack(state.state_vector[1, 0] for state in measurements),
    np.hstack(state.state_vector[0, 0] for state in measurements))
ax.scatter(x + sensor_x,
           y + sensor_y,
           color='b')
fig

plt.polar([state.state_vector[0, 0] for state in measurements], 
        [state.state_vector[1, 0] for state in measurements])

# Create Models and Extended Kalman Filter

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.1), ConstantVelocity(0.1)))

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
measurement_model = CartesianToBearingRange(
    4, # Number of state dimensions (position and velocity in 2D)
    (0,2), # Mapping measurement vector index to state index
    np.diag([np.radians(0.2), 1]),  # Covariance matrix for Gaussian PDF
    translation_offset=np.array([[sensor_x], [sensor_y]]) # Location of sensor in cartesian.
)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model)

# Running the Extended Kalman Filter

from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1, 1, 1, 1]), timestamp=start_time)

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement) # Used to group a prediction and measurement together
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# Plot the resulting track
ax.plot([state.state_vector[0, 0] for state in track], 
        [state.state_vector[2, 0] for state in track],
        marker=".")
fig

from matplotlib.patches import Ellipse
HH = np.array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.]])
for state in track:
    w, v = np.linalg.eig(HH@state.covar@HH.T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0,2), 0],
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)
fig

