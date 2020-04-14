# -*- coding: utf-8 -*-
# CREATING TRACKERS USING STONE SOUP
# Single Target Detection Simulator Example.

# 1. Setup Script

#General imports and plotting
import datetime
import numpy as np

# Plotting
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-colorblind')

# 2. Generate Data 

#MODELS 
# Linear Gaussian transition model like in the 01-Kalman example
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(1), ConstantVelocity(1)))

# Use a Linear Gaussian measurement model like in 01-Kalman example
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state = 4, # Number of state dimensions (position and velocity in 2D)
    mapping = [0,2], # Mapping measurement vector index to state index
    noise_covar = np.diag([10, 10])  # Covariance matrix for Gaussian PDF
    )

#SIMULATORS
# GROUND TRUTH SIMULATOR

# Before running the Detection Simulator, a Ground Truth reader/simulator must be used in order to generate 
# Ground Truth Data which is used as an input to the Detection Simulator.
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix

# Arbitrary initial state of target.
initial_state=GaussianState( StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([1000000, 10, 1000000, 10])))

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
groundtruth_sim = SingleTargetGroundTruthSimulator(
            transition_model=transition_model,
            initial_state=initial_state,
            timestep=datetime.timedelta(seconds=5),
            number_steps=100)

# DETECTION SIMULATOR

from stonesoup.simulator.simple import SimpleDetectionSimulator
detection_sim = SimpleDetectionSimulator(groundtruth=groundtruth_sim,
                       measurement_model=measurement_model,
                       meas_range=np.array([[-1, 1],[-1, 1]])*5000, 
                       detection_probability=0.9, 
                       clutter_rate=1)
                                
detections_source = detection_sim

# 3. Implement Feeder 
# Not using a feeder


# 4. Set up Tracker Components


# Kalman Filter Components 
# PRIOR STATE
# Creating a prior estimate of where we think our target will be
#prior = initial_state 

# PREDICTOR
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# UPDATER 
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# Data Association Components
# HYPOTHESIER
# Mahalanobis Hypothesiser as in 04-Data Association
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

#DATAASSOCIATOR
from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

# Computer Components

# INITIATOR
from stonesoup.initiator.simple import SimpleMeasurementInitiator
initiator = SimpleMeasurementInitiator(
    GaussianState(np.array([[0], [0], [0], [0]]), np.diag([1000000, 10, 1000000, 10])),
    measurement_model=measurement_model)

# DELETER
from stonesoup.deleter.error import CovarianceBasedDeleter
deleter = CovarianceBasedDeleter(covar_trace_thresh=1E3)


# 5. Running the Tracker

# Single Target Tracker

from stonesoup.tracker.simple import SingleTargetTracker
tracker = SingleTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detections_source,
    data_associator=data_associator,
    updater=updater,
)


# 6. Display Results

tracks = set()
#for time,current_tracks in tracker.tracks_gen():
#    tracks = tracks.union(current_tracks)
groundtruth_paths = set()  # Store for plotting later
detections = set()  # Store for plotting later

for step, (time,ctracks) in enumerate(tracker.tracks_gen(),1):
    tracks.update(ctracks)
    detections |= tracker.detector.detections
    if not step % 10:
       print("Step: {} Time: {}".format(step, time))

# METRICSGENERATOR 
from stonesoup.metricgenerator.plotter import TwoDPlotter
TwoDPlotter([0, 1], [0, 1], [0, 1]).plot_tracks_truth_detections(tracks, groundtruth_paths, detections)
