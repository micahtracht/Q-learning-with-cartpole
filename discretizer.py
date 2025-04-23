import numpy as np

class Discretizer:
    def __init__(self, bins_per_feature, lower_bounds, upper_bounds):
        # For each low, high in the feature range, np.linspace makes evenly spaced cut points. Using bins_per_feature + 1 points, we divide the range into bins_per_feature segments.
        # [1:-1] cuts out the endpoints, leaving only the internal boundaries. This is important for later when we use np.digitize, as it expects internal boundaries (bin edges), NOT endpoints.
        # It also avoids issues like values falling on the mins, maxes, out-of-bounds (invalid) indices, or on edges of bins.
        self.bins = [np.linspace(low, high, bins_per_feature + 1)[1:-1]
        for low, high in zip(lower_bounds, upper_bounds)]
    
    
    # This is used to convert a continuous state vector (eg, a 4 dim tuple representing our cart pos, pole angle, cart velocity, and pole angular velocity) into discrete bins.
    # That state vector is our 'observation', and is a 4-tuple or list with len=4.
    # We use np.digitize(value, bin_edges) to do it. This takes in the value, which can be continuous (and should be), and our bin edges, and returns the index of the bin that our value belongs in.
    # We'll use self.bins, as it contains a separate array of bin edges for each feature, based on it's value range.
    # Note that if we have a value below our first bin edge, it maps to 0, and if it's above our last bin edge, it maps to n_bins - 1. This is good, as it avoids index errors (this is why we used [1:-1 above])
    def discretize(self, observation):
        # match each feature value with its bin using .zip (zip(observation, self.bins))
        # digitize them using np.digitize to get the index of the bin they belong to (np.digitize(feature_val, bin_edges))
        # collect these indices into our 'state' vector, which is our discretized original state. (state = tuple(...))
        state = tuple(np.digitize(feature_val, bin_edges) for feature_val, bin_edges in zip(observation, self.bins))
        return state