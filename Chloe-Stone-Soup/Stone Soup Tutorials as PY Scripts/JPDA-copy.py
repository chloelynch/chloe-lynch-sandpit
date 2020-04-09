# -*- coding: utf-8 -*-
# Stone Soup JPDA Example

#General imports and plotting
import datetime
import numpy as np

# Plotting
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-colorblind')

# Generating Data
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel,\
                                               ConstantVelocity
transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantVelocity(1), ConstantVelocity(1)))

from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4, mapping=[0, 2], noise_covar=np.diag([10, 10]))

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix

groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([1000000, 10, 1000000, 10]))),
    timestep=datetime.timedelta(seconds=5),
    birth_rate=0.3,
    death_probability=0.05
)

from stonesoup.simulator.simple import SimpleDetectionSimulator

detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    meas_range=np.array([[-1, 1], [-1, 1]])*5000,  # Area to generate clutter
    detection_probability=0.9,
    clutter_rate=3,
)

detections_source = detection_sim

# Building JPDA Kalman Tracker Components 

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor, updater, clutter_spatial_density=detection_sim.clutter_spatial_density, prob_detect=0.9, prob_gate=0.99)

from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser, 0.85)

from stonesoup.initiator.simple import SinglePointInitiator
initiator = SinglePointInitiator(
    GaussianState(np.array([[0], [0], [0], [0]]), np.diag([10000, 100, 10000, 1000])),
    measurement_model=measurement_model)

from stonesoup.deleter.error import CovarianceBasedDeleter
deleter = CovarianceBasedDeleter(covar_trace_thresh=1E3)

# Running the JPDA Kalman Tracker 

from stonesoup.tracker.simple import MultiTargetMixtureTracker
tracker = MultiTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detections_source,
    data_associator=data_associator,
    updater=updater,
)

tracks = set()
groundtruth_paths = set()  # Store for plotting later
detections = set()  # Store for plotting later
for step, (time, ctracks) in enumerate(tracker.tracks_gen(), 1):
    tracks.update(ctracks)
    detections |= tracker.detector.detections
    if not step % 10:
        print("Step: {} Time: {}".format(step, time))
        
        from stonesoup.metricgenerator.plotter import TwoDPlotter
TwoDPlotter([0, 2], [0, 2], [0, 1]).plot_tracks_truth_detections(tracks, groundtruth_paths, detections)


