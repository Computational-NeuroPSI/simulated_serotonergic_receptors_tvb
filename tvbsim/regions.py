import numpy as np
import os
from tvb.basic.readers import ZipReader

def get_distances_to_region(parameters, region_id):
    """
        Returns an array of size n_region giving the distance of each region wrt region_id
    """
    reader = ZipReader(os.path.join(
                        parameters.parameter_connection_between_region['path'],
                        parameters.parameter_connection_between_region['conn_name']))
    centers = reader.read_array_from_file('centres.txt', use_cols=(1,2,3))
    return np.linalg.norm(centers[region_id] - centers, axis=1)

def get_regions_within_distance(general_parameters, region_id, distance):
    """
        Given a region_id, returns all the regions such that their centers is within a sphere of radius distance around
        the center of region with id region_id
    """
    distances_to_region = get_distances_to_region(general_parameters, region_id)
    idx_closest_sorted = np.argsort(distances_to_region)
    return idx_closest_sorted[distances_to_region[idx_closest_sorted] < distance]

def get_closest_regions(parameters, region_id, n):
    """
        Returns the n regions that are closest to the region whose id is given
    """
    distances_to_region = get_distances_to_region(parameters, region_id)
    idx_closest_sorted = np.argsort(distances_to_region)[:n]
    closest_regions = idx_closest_sorted.tolist() # conversion this way necessary for good JSON serialization of parameters
    return closest_regions

# TODO : maybe better way to do this if hemisphere.txt exists
def hemisphere_regions(parameters, hemisphere):
    """
        Returns regions of the chosen hemisphere ('l' or 'r') by their name (ends or starts with l/r)
    """
    assert(hemisphere == 'l' or 'r'), "For left hemisphere : 'l', For right hemisphere : 'r'"
    reader = ZipReader(os.path.join(
                        parameters.parameter_connection_between_region['path'],
                        parameters.parameter_connection_between_region['conn_name']))
    centers = reader.read_array_from_file('centres.txt', use_cols=(0,), dtype=str)
    return [i for i,center in enumerate(centers) 
            if center.lower().endswith(hemisphere) or center.lower().startswith(hemisphere)]
def get_left_hemisphere_regions(parameters):
    """
        Returns left hemisphere regions
    """
    return hemisphere_regions(parameters, 'l')
def get_right_hemisphere_regions(parameters):
    """
        Returns right hemisphere regions
    """
    return hemisphere_regions(parameters, 'r')

def get_connected_regions(parameters, region_id):
    """
        Get regions that have a non-zero connection (in or out) with the given region
    """
    reader = ZipReader(os.path.join(
                        parameters.parameter_connection_between_region['path'],
                        parameters.parameter_connection_between_region['conn_name']))
    weights = reader.read_array_from_file('weights.txt')
    incoming = np.where(weights[region_id] != 0)[0]
    outgoing = np.where(weights[:,region_id] != 0)[0]
    return set(region_id + [*incoming, *outgoing])
        
#def get_most_connected_regions(parameters, region_id, n_regions, direction):
#    """
#        Get regions that have a non-zero connection (in given direction, 'in' or 'out') with the given region.
#    """
#    # weights[region_id] : ingoing or outgoing ?
#    reader = ZipReader(os.path.join(
#                        parameters.parameter_connection_between_region['path'],
#                        parameters.parameter_connection_between_region['conn_name']))
#    weights = reader.read_array_from_file('weights.txt')
#    if direction == 'in':
#        weights = weights[:,region_id]
#    elif direction == 'out':
#        weights = weights[region_id]
#    return [region_id] + list(np.argsort(-np.abs(weights))[:n_regions])