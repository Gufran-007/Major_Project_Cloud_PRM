"""
cloud-prm — Predictive Resource Management for Cloud Data Centres.

Implementation of the Phase-II design: forecasting pipeline, allocation
policies, and a trace-replay simulator for comparing them.

Package layout (read them in this order the first time):

    config.py           loads the YAML experiment configuration
    synthetic.py        makes a fake Bitbrains-like trace so you can run with no data
    data.py             loads + cleans + resamples real or synthetic traces
    features.py         turns a clean time series into a supervised learning table
    splits.py           chronological train/validation/test splitting
    models.py           every forecaster, all behind one common interface
    metrics.py          error metrics, reported separately for over/under-prediction
    residuals.py        rolling-residual dispersion -> the allocation safety margin
    forecast_service.py the frozen forecaster -> allocator contract
    experiment.py       wires all of the above together and writes result tables
"""
__version__ = "1.0.0"
