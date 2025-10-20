"""
The following code was utmost politely borrowed and adapted from
https://github.com/JFvdH/Efficient-SD-through-AE/blob/main/beamSearch.py
It came with the following header docstring:
    The following code was adapted from W. Duivesteijn, T.C. van Dijk. (2021)
    Exceptional Gestalt Mining: Combining Magic Cards to Make Complex Coalitions Thrive.
    In: Proceedings of the 8th Workshop on Machine Learning and Data Mining for Sports Analytics.
    Available from http://wwwis.win.tue.nl/~wouter/Publ/J05-EMM_DMKD.pdf
"""

# Package imports
import heapq
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Classes
class BoundedPriorityQueue:
    """Used to store the <q> most promising subgroups, ensures uniqueness.
    Keeps a maximum size (throws away value with least quality).
    """

    def __init__(self, bound):
        # Initializes empty queue with maximum length of <bound>
        self.values = []
        self.bound = bound
        self.entry_count = 0

    def add(self, element, quality, **adds):
        # Adds <element> to the bounded priority queue if it is of sufficient quality
        new_entry = (quality, self.entry_count, element, adds)
        if len(self.values) >= self.bound:
            heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    def get_values(self):
        # Returns elements in bounded priority queue in sorted order
        for (q, _, e, x) in sorted(self.values, reverse=True):
            yield q, e, x

    def show_contents(self):
        # Prints contents of the bounded priority queue (used for debugging)
        print("show_contents")
        for (q, entry_count, e) in self.values:
            print(q, entry_count, e)


class Queue:
    """Used to store candidate solutions, ensures uniqueness."""

    def __init__(self):  # Initializes empty queue
        self.items = []

    def is_empty(self):  # Returns True if queue is empty, False otherwise
        return self.items == []

    def enqueue(self, item):  # Adds <item> to queue if it is not already present
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self):  # Pulls one item from the queue
        return self.items.pop()

    def size(self):  # Returns the number of items in the queue
        return len(self.items)

    def get_values(self):  # Returns the queue (as a list)
        return self.items

    def add_all(self, iterable):  # Adds all items in <iterable> to the queue, given they are not already present
        for item in iterable:
            self.enqueue(item)

    def clear(self):  # Removes all items from the queue
        self.items.clear()


# Functions
def refine(desc, more):
    # Creates a copy of the seed <desc> and adds it to the new selector <more>
    # Used to prevent pointer issues with selectors
    copy = desc[:]
    copy.append(more)
    return copy


def as_string(desc):
    # Adds " and " to <desc> such that selectors are properly separated when the refine function is used
    return " and ".join(desc)


def eta(seed, df, features, n_chunks=5):
    # Returns a generator which includes all possible refinements of <seed> for the given <features> on dataset <df>
    # n_chunks refers to the number of possible splits we consider for numerical features

    print("eta ", seed)
    if seed:  # we only specify more on the elements that are still in the subset
        d_str = as_string(seed)
        ind = df.eval(d_str)
        df_sub = df.loc[ind,]
    else:
        df_sub = df
    for f in features:
        column_data = pd.DataFrame()

        if (df_sub[f].dtype == "float64") or (df_sub[f].dtype == "float32"):
            # get quantiles here instead of intervals for the case that data are very skewed
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1, n_chunks + 1):  # determine the number of chunks you want to divide your data in
                x = np.percentile(dat, 100 / i)  #
                candidate = "{} <= {}".format(f, x)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
        elif df_sub[f].dtype == "object":
            column_data = df_sub[f]
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
        elif df_sub[f].dtype == "int64":
            column_data = df_sub[f]
            dat = np.sort(column_data)
            dat = dat[np.logical_not(np.isnan(dat))]
            for i in range(1, n_chunks + 1):  # determine the number of chunks you want to divide your data in
                x = np.percentile(dat, 100 / i)  #
                candidate = "{} <= {}".format(f, x)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} > {}".format(f, x)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
        elif df_sub[f].dtype == "bool":
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
                candidate = "{} != '{}'".format(f, i)
                if candidate not in seed:  # if not already there
                    yield refine(seed, candidate)
        else:
            assert False


# def eval_quality(desc, df, target):
#     # Function used to calculate the solution"s WRAcc
#     sub_group = df[df.eval(as_string(desc))]
#     prop_p_sg = len(sub_group[sub_group[target] == 1]) / len(sub_group)
#     prop_p_df = len(df[df[target] == 1]) / len(df)
#     wracc = ((len(sub_group) / len(df)) ** 1) * (prop_p_sg - prop_p_df)  # for WRAcc a=1
#     return wracc


def satisfies_all(desc, df, threshold=0.05):
    """Checks if a subgroup is of a good size - not too small, not too big."""
    d_str = as_string(desc)
    ind = df.eval(d_str)
    return len(df) * threshold <= sum(ind) <= len(df) * (1 - threshold)


def mahalanobis_quality(desc, df, eeg_features, binary_target):
    """Calculates a composite quality score for a subgroup.
    The score combines:
    1. WRAcc: To measure how exceptional the proportion of unhealthy patients is.
    2. Mahalanobis Distance: To measure how much the subgroup's average EEG features
       deviate from the complement average, accounting for covariance.
    A high score requires a subgroup to be exceptional in both aspects.
    """
    # Get subgroup and complement
    sub_group = df[df.eval(as_string(desc))]
    sub_complement = df[~df.eval(as_string(desc))]

    # Handle edge cases
    if len(sub_group) == 0 or len(sub_complement) == 0:
        return 0.0

    # Calculate entropy (phi_ef)
    p_n = len(sub_group) / len(df)
    p_n_c = len(sub_complement) / len(df)
    entropy = -p_n * np.log2(p_n) - p_n_c * np.log2(p_n_c)
    # Handle floating point issues if p_n is near 0 or 1
    if np.isnan(entropy):
        entropy = 0.0

    # Calculate WRAcc for the binary target (is_unhealthy)
    prop_p_sg = sub_group[binary_target].mean()
    prop_p_df = df[binary_target].mean()
    wracc = (len(sub_group) / len(df)) * (prop_p_sg - prop_p_df)
    if wracc <= 0:
        return 0.0

    # Calculate the Mahalanobis distance for the EEG features model
    sub_mean_eeg = sub_group[eeg_features].mean().values
    sub_complement_mean_eeg = sub_complement[eeg_features].mean().values
    cov_matrix = sub_complement[eeg_features].cov().values
    regularization_factor = 1e-6
    inv_cov_matrix_eeg = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_factor)

    # The distance measures how many standard deviations away the subgroup mean is from the global mean.
    mahalanobis_dist = mahalanobis(sub_mean_eeg, sub_complement_mean_eeg, inv_cov_matrix_eeg)

    with open("tmp.csv", "a") as f:
        f.write(f"{wracc}, {mahalanobis_dist}\n")

    # Return the composite quality score
    return entropy * wracc * mahalanobis_dist


def regression_quality(desc, df, eeg_features, binary_target):
    """Calculates a quality score based on entropy and logistic regression slope difference.
    It combines:
    1. Entropy (phi_ef): Measures the informativeness of the
       split created by the subgroup and its complement.
    2. Model Difference: The absolute difference in the slope coefficient
       of a simple logistic regression model fitted on the subgroup and the complement.
    A high score requires a subgroup to be exceptional in both aspects.
    """
    # Get subgroup and complement
    sub_group = df[df.eval(as_string(desc))]
    sub_complement = df[~df.eval(as_string(desc))]

    # Handle edge cases
    if len(sub_group) == 0 or len(sub_complement) == 0:
        return 0.0

    # Define X (inputs) and Y (output)
    X_g = sub_group[eeg_features]
    y_g = sub_group[binary_target]
    X_c = sub_complement[eeg_features]
    y_c = sub_complement[binary_target]

    # We cannot fit a logistic regression if either group has only one class
    if len(np.unique(y_g)) < 2 or len(np.unique(y_c)) < 2:
        return 0.0

    # Calculate entropy (phi_ef)
    p_n = len(sub_group) / len(df)
    p_n_c = len(sub_complement) / len(df)
    entropy = -p_n * np.log2(p_n) - p_n_c * np.log2(p_n_c)
    # Handle floating point issues if p_n is near 0 or 1
    if np.isnan(entropy):
        entropy = 0.0

    # Calculate WRAcc
    prop_p_sg = sub_group[binary_target].mean()
    prop_p_df = df[binary_target].mean()
    wracc = (len(sub_group) / len(df)) * (prop_p_sg - prop_p_df)
    if wracc <= 0:
        return 0.0

    # Calculate regression coefficients
    try:
        # Fit the scaler on the entire dataset to ensure G and C are scaled consistently.
        scaler = StandardScaler().fit(df[eeg_features])
        X_g_scaled = scaler.transform(X_g)
        X_c_scaled = scaler.transform(X_c)

        # Fit logistic regression on subgroup
        model_g = LogisticRegression(solver="liblinear", class_weight="balanced").fit(X_g_scaled, y_g)
        beta_g = model_g.coef_[0]

        # Fit logistic regression on complement
        model_c = LogisticRegression(solver="liblinear", class_weight="balanced").fit(X_c_scaled, y_c)
        beta_c = model_c.coef_[0]

    except Exception:
        return 0.0

    # Calculate Model Difference
    model_diff = np.linalg.norm(beta_g - beta_c)

    # Return the composite quality score
    return entropy * wracc * model_diff


def EMM(w, d, q, catch_all_description, df, features, eeg_features,
        target, n_chunks=5, ensure_diversity=False, quality_name="regression"):
    """EMM main loop.
    w - width of beam, i.e. the max number of results in the beam
    d - num levels, i.e. how many attributes are considered
    q - max results, i.e. max number of results output by the algorithm
    eta - function that receives a description and returns all possible refinements
    satisfies_all - function that receives a description and verifies wheather it satisfies some requirements as needed
    eval_quality - returns quality for a given description. This should be comparable to qualities of other descriptions
    catch_all_description - the equivalent of True, or all, as that the whole dataset shall match
    df - dataframe of mined dataset
    features - descriptive features for subgroup discovery
    eeg_features - numeric features on which the model is built
    target - column name of target attribute in df
    quality_name - quality function name
    """
    # Initialize variables
    result_set = BoundedPriorityQueue(q)  # Set of results, can contain results from multiple levels
    candidate_queue = Queue()  # Set of candidate solutions to consider adding to the result_set
    candidate_queue.enqueue(catch_all_description)  # Set of results on a particular level
    error = 0.00001  # Allowed error margin (due to floating point error) when comparing the quality of solutions

    quality_f = None
    match quality_name:
        case "regression":
            quality_f = regression_quality
        case "mahalanobis":
            quality_f = mahalanobis_quality
        case _:
            raise ValueError(f"Unknown quality_name: {quality_name}.")

    # Perform BeamSearch for <d> levels
    for level in range(d):
        print("level : ", level)

        # Initialize this level's beam
        beam = BoundedPriorityQueue(w)

        # Go over all rules generated on previous level, or "empty" rule if level = 0
        for seed in candidate_queue.get_values():
            print("    seed : ", seed)

            # Start by evaluating the quality of the seed
            if seed:
                seed_quality = quality_f(seed, df, eeg_features, target)
            else:
                seed_quality = 99

            # For all refinements created by eta function on descriptions (i.e. features),
            # which can be different types of columns
            # eta(seed) reads the dataset given certain seed (i.e. already created rules) and looks at new descriptions
            for desc in eta(seed, df, features, n_chunks):

                # Check if the subgroup contains at least x% of data, proceed if yes
                if satisfies_all(desc, df):

                    # Calculate the new solution's quality
                    quality = quality_f(desc, df, eeg_features, target)

                    # Ensure diversity by forcing difference in quality when compared to its seed
                    # if <ensure_diversity> is set to True. Principle is based on:
                    # Van Leeuwen, M., & Knobbe, A. (2012), Diverse subgroup set discovery.
                    # Data Mining and Knowledge Discovery, 25(2), 208-242.
                    if ensure_diversity:
                        if quality < (seed_quality * 1 - error) or quality > (seed_quality * 1 + error):
                            result_set.add(desc, quality)
                            beam.add(desc, quality)
                    else:
                        result_set.add(desc, quality)
                        beam.add(desc, quality)

        # When all candidates for a search level have been explored,
        # the contents of the beam are moved into candidate_queue, to generate next level candidates
        candidate_queue = Queue()
        candidate_queue.add_all(desc for (_, desc, _) in beam.get_values())

    # Return the <result_set> once the BeamSearch algorithm has completed
    return result_set
