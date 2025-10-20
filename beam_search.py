"""
The following code was adapted from W. Duivesteijn, T.C. van Dijk. (2021)
Exceptional Gestalt Mining: Combining Magic Cards to Make Complex Coalitions Thrive.
In: Proceedings of the 8th Workshop on Machine Learning and Data Mining for Sports Analytics.
Available from http://wwwis.win.tue.nl/~wout

Modified to use regression models for Exceptional Model Mining instead of mean-based models.
"""

import heapq
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class BoundedPriorityQueue:
    """Used to store candidate solutions, as to narrow down to the most promising subgroups,
    ensures uniqueness. Keeps a maximum size (throws away value with least quality)."""

    def __init__(self, bound):
        self.values = []
        self.bound = bound
        self.entry_count = 0

    def add(self, element, quality, **adds):
        """Adds to the bounded priority queue if it is of sufficient quality"""
        new_entry = (quality, self.entry_count, element, adds)
        if len(self.values) >= self.bound:
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)
        self.entry_count += 1

    def get_values(self):
        """Returns elements in bounded priority queue in sorted order"""
        for (q, _, e, x) in sorted(self.values, reverse=True):
            yield q, e, x

    def show_contents(self):
        """Prints contents of the bounded priority queue (used for debugging)"""
        print("show_contents")
        for (q, entry_count, e) in self.values:
            print(q, entry_count, e)


class Queue:
    """Used to store candidate solutions, ensures uniqueness."""

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        """Adds item to queue if it is not already present"""
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self):
        """Pulls one item from the queue"""
        return self.items.pop()

    def size(self):
        """Returns the number of items in the queue"""
        return len(self.items)

    def get_values(self):
        """Returns the queue (as a list)"""
        return self.items

    def add_all(self, iterable):
        """Adds all items to the queue, given they are not already present"""
        for item in iterable:
            self.enqueue(item)

    def clear(self):
        """Removes all items from the queue"""
        self.items.clear()


# Helper functions
def refine(desc, more):
    """Creates a copy of the seed and adds it to the new selector"""
    copy = desc[:]
    copy.append(more)
    return copy


def as_string(desc):
    """Adds " and " such that selectors are properly separated"""
    return " and ".join(desc)


def eta(seed, df, features, n_chunks=5):
    """Returns a generator which includes all possible refinements for the given seed on dataset
    n_chunks refers to the number of possible splits we consider for numerical features"""

    if seed:
        d_str = as_string(seed)
        ind = df.eval(d_str)
        df_sub = df.loc[ind,]
    else:
        df_sub = df

    for f in features:
        if (df_sub[f].dtype == "float64") or (df_sub[f].dtype == "float32"):
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]

            for i in range(1, n_chunks + 1):
                x = np.percentile(dat, 100 / i)
                candidate = f"{f} <= {x}"
                if candidate not in seed:
                    yield refine(seed, candidate)
                candidate = f"{f} > {x}"
                if candidate not in seed:
                    yield refine(seed, candidate)

        elif df_sub[f].dtype == "object":
            column_data = df_sub[f]
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = f"{f} == '{i}'"
                if candidate not in seed:
                    yield refine(seed, candidate)
                candidate = f"{f} != '{i}'"
                if candidate not in seed:
                    yield refine(seed, candidate)

        elif df_sub[f].dtype == "int64":
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]

            for i in range(1, n_chunks + 1):
                x = np.percentile(dat, 100 / i)
                candidate = f"{f} <= {x}"
                if candidate not in seed:
                    yield refine(seed, candidate)
                candidate = f"{f} > {x}"
                if candidate not in seed:
                    yield refine(seed, candidate)

        elif df_sub[f].dtype == "bool":
            uniq = df_sub[f].dropna().unique()
            for i in uniq:
                candidate = f"{f} == '{i}'"
                if candidate not in seed:
                    yield refine(seed, candidate)
                candidate = f"{f} != '{i}'"
                if candidate not in seed:
                    yield refine(seed, candidate)


def satisfies_all(desc, df, threshold=0.2):
    """Checks if a subgroup is of a good size - not too small, not too big."""
    d_str = as_string(desc)
    ind = df.eval(d_str)
    return len(df) * threshold <= sum(ind) <= len(df) * (1 - threshold)


def eval_quality_regression(desc, df, target_features, binary_target, quality_metric="cooks_distance"):
    """Calculates a quality score for a subgroup based on regression model exceptionality.

    Parameters:
    -----------
    desc : list
        Description defining the subgroup
    df : DataFrame
        The complete dataset
    target_features : list
        Features to use in the regression model (predictors and response)
    binary_target : str
        Binary target for WRAcc calculation
    quality_metric : str
        Which metric to use: "cooks_distance", "slope_difference", "r2_difference", 
        "mse_ratio", "prediction_difference"

    Returns:
    --------
    float : Quality score (higher is better)
    """
    # Get subgroup and complement
    if not desc:
        return 0.0

    sub_group = df[df.eval(as_string(desc))]
    sub_complement = df[~df.eval(as_string(desc))]

    if len(sub_group) < 10 or len(sub_complement) < 10:  # Minimum size for reliable regression
        return 0.0

    # Calculate WRAcc for the binary target as a relevance weight
    prop_p_sg = sub_group[binary_target].mean()
    prop_p_df = df[binary_target].mean()
    wracc = (len(sub_group) / len(df)) * (prop_p_sg - prop_p_df)

    # Only consider subgroups with exceptional binary target distribution
    if wracc <= 0:
        return 0.0

    # Prepare regression data (assume last feature is the response variable)
    X_features = target_features[:-1]
    y_feature = target_features[-1]

    # Fit regression models
    try:
        # Global model
        X_global = df[X_features].values
        y_global = df[y_feature].values
        model_global = LinearRegression()
        model_global.fit(X_global, y_global)
        beta_global = np.concatenate([[model_global.intercept_], model_global.coef_])

        # Subgroup model
        X_sg = sub_group[X_features].values
        y_sg = sub_group[y_feature].values
        model_sg = LinearRegression()
        model_sg.fit(X_sg, y_sg)
        beta_sg = np.concatenate([[model_sg.intercept_], model_sg.coef_])

        # Complement model
        X_comp = sub_complement[X_features].values
        y_comp = sub_complement[y_feature].values
        model_comp = LinearRegression()
        model_comp.fit(X_comp, y_comp)
        beta_comp = np.concatenate([[model_comp.intercept_], model_comp.coef_])

    except Exception as e:
        return 0.0

    # Calculate quality based on chosen metric
    n_sg = len(sub_group)
    n_total = len(df)
    p = len(beta_global)

    if quality_metric == "cooks_distance":
        # Cook\'s Distance: measures influence of subgroup on global model
        # Based on Duivesteijn et al. 2012
        y_pred_global = model_global.predict(X_global)
        y_pred_sg = model_sg.predict(X_global)
        s_squared = np.sum((y_global - y_pred_global)**2) / (n_total - p)

        if s_squared < 1e-10:
            return 0.0

        cooks_d = np.sum((y_pred_sg - y_pred_global)**2) / (p * s_squared)
        quality = (n_sg / n_total) * cooks_d

    elif quality_metric == "slope_difference":
        # Difference in regression coefficients (slopes)
        # Weighted by subgroup size
        slope_diff = np.linalg.norm(beta_sg - beta_global)
        quality = (n_sg / n_total) * slope_diff

    elif quality_metric == "slope_difference_complement":
        # Difference between subgroup and complement models
        # This is the EMM-preferred approach
        slope_diff = np.linalg.norm(beta_sg - beta_comp)
        quality = (n_sg / n_total) * slope_diff

    elif quality_metric == "r2_difference":
        # Difference in R² between subgroup and global model
        r2_global = model_global.score(X_global, y_global)
        r2_sg = model_sg.score(X_sg, y_sg)
        r2_diff = abs(r2_sg - r2_global)
        quality = (n_sg / n_total) * r2_diff

    elif quality_metric == "mse_ratio":
        # Ratio of MSE in subgroup vs complement
        y_pred_sg = model_sg.predict(X_sg)
        y_pred_comp = model_comp.predict(X_comp)
        mse_sg = mean_squared_error(y_sg, y_pred_sg)
        mse_comp = mean_squared_error(y_comp, y_pred_comp)

        if mse_comp < 1e-10:
            return 0.0

        mse_ratio = abs(np.log(mse_sg / mse_comp + 1e-10))
        quality = (n_sg / n_total) * mse_ratio

    elif quality_metric == "prediction_difference":
        # Average difference in predictions between subgroup and global model
        y_pred_global_sg = model_global.predict(X_sg)
        y_pred_sg = model_sg.predict(X_sg)
        pred_diff = np.mean((y_pred_sg - y_pred_global_sg)**2)
        quality = (n_sg / n_total) * pred_diff

    else:
        raise ValueError(f"Unknown quality metric: {quality_metric}")

    # Multiply by WRAcc to ensure clinical relevance
    return wracc * quality


def EMM_regression(w, d, q, catch_all_description, df, descriptive_features, 
                   target_features, binary_target, quality_metric="cooks_distance",
                   n_chunks=5, ensure_diversity=False):
    """EMM main loop using regression models.

    Parameters:
    -----------
    w : int
        Width of beam (max number of results in the beam)
    d : int
        Number of levels (how many attributes are considered)
    q : int
        Max results (max number of results output by the algorithm)
    catch_all_description : list
        The equivalent of True, or all, such that the whole dataset shall match
    df : DataFrame
        Dataframe of mined dataset
    descriptive_features : list
        Features for subgroup discovery
    target_features : list
        Numeric features on which the regression model is built
    binary_target : str
        Column name of binary target attribute in df
    quality_metric : str
        Which regression quality metric to use
    n_chunks : int
        Number of possible splits for numerical features
    ensure_diversity : bool
        Whether to ensure diversity in results

    Returns:
    --------
    BoundedPriorityQueue : Result set containing top-q subgroups
    """
    result_set = BoundedPriorityQueue(q)
    candidate_queue = Queue()
    candidate_queue.enqueue(catch_all_description)
    error = 0.00001

    for level in range(d):
        print(f"Level: {level}")
        beam = BoundedPriorityQueue(w)

        for seed in candidate_queue.get_values():
            print(f"  Seed: {seed}")

            # Evaluate seed quality
            if seed:
                seed_quality = eval_quality_regression(seed, df, target_features, 
                                                      binary_target, quality_metric)
            else:
                seed_quality = 99

            # Explore refinements
            for desc in eta(seed, df, descriptive_features, n_chunks):
                if satisfies_all(desc, df):
                    quality = eval_quality_regression(desc, df, target_features, 
                                                     binary_target, quality_metric)

                    if ensure_diversity:
                        if quality < (seed_quality * (1 - error)) or quality > (seed_quality * (1 + error)):
                            result_set.add(desc, quality)
                            beam.add(desc, quality)
                    else:
                        result_set.add(desc, quality)
                        beam.add(desc, quality)

        # Move beam to candidate queue for next level
        candidate_queue = Queue()
        candidate_queue.add_all(e for (q, e, x) in beam.get_values())

    return result_set
