import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

DEBUG_MODE = False

def prepare_data(df, target_col, exog_cols):
    """
    Prepare and align target and exogenous data, drop rows with missing values.
    """
    if exog_cols:
        combined = pd.concat([df[target_col], df[exog_cols]], axis=1)
        combined = combined.dropna()
        y = combined[target_col]
        exog = combined[exog_cols]
    else:
        y = df[target_col].dropna()
        exog = None
    return y, exog

def create_sarimax(params, y, exog):
    """
    Create SARIMAX model object.
    """
    model = SARIMAX(
        y,
        exog=exog,
        order=(params['p'], params['d'], params['q']),
        seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model

def evaluate_params(params, df, target_col, exog_cols):
    """
    Evaluate parameter set by fitting SARIMAX and returning AIC.
    """
    try:
        y, exog = prepare_data(df, target_col, exog_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = create_sarimax(params, y, exog)
            results = model.fit(disp=False)
            return results.aic
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error fitting params {params}: {str(e)}")
        return np.inf

def fit_best_model(best_params, df, target_col, exog_cols):
    """
    Fit SARIMAX model with best parameters.
    """
    y, exog = prepare_data(df, target_col, exog_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = create_sarimax(best_params, y, exog)
        results = model.fit(disp=False)
    return results
