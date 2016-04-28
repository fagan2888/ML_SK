import os
import hashlib
import collections

import ML_SK.parser.common

def convert_to_md5(data_filenames, settings):
    """
    Converts a string based on settings to md5
    This way we can avoid parsing the same data
    over and over again.
    """
    string = str(data_filenames)
    string += str(settings.__dict__)

    return hashlib.md5(string).hexdigest()


def parse_data(data_filenames, settings):
    '''
    Checks if these settings have been run before, and opens a saved pickle if it was.
    Else parses xyz files with atomic properties in the 5th through the last column and molecular properties
    in the second line. Afterwards constructs the descriptor.
    Returns a tuple a,b,c where a is a 2d ndarray of size (N_elements,descriptor_length),
    b is a 1d array of the property and c is a dictionary that maps each molecule with the atoms in it.
    Finally saves the data in a pickle.
    '''

    md5 = convert_to_md5(data_filenames, settings)

    # if pickle already exist, then load it.
    pickle_path = settings.output_folder + "/" + md5 + ".pickle"
    if os.path.isfile(pickle_path):
        if settings.verbose:
            print "Loading pickle"
        descriptor, observables, molecule_indices = \
            ML_SK.parser.common.load_parsed_data(pickle_path)
        if settings.verbose:
            print "Loaded {} molecules having {} {}-atoms".format(
                    len(molecule_indices),descriptor.shape[0], settings.target_element)
        return descriptor, observables, molecule_indices

    if settings.verbose:
        print 'Parsing data'

    descriptor, observables, molecule_indices = \
            ML_SK.parser.common.parse_atomic_data(data_filenames, settings)

    if settings.verbose:
        print "Parsed {} molecules having {} {}-atoms".format(
                len(molecule_indices),descriptor.shape[0], settings.target_element)

    if settings.verbose:
        print "Saving pickle"
    ML_SK.parser.common.save_parsed_data((descriptor, observables, molecule_indices),pickle_path)

    return descriptor, observables, molecule_indices
