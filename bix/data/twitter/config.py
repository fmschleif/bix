"""
This is the Config file for the Twitter-Module.

The 'Twitter_*' keys in the TWITTER_CONFIG dictionary must be set to a valid configuration for this module to work.
Alternatively this values can be set as environment Variables or specified when instantiating the TwitterRetriever
class.
"""

import os


TWITTER_CONFIG = {
    'TWITTER_ACCESS_TOKEN': None,
    'TWITTER_ACCESS_SECRET': None,
    'TWITTER_CONSUMER_TOKEN': None,
    'TWITTER_CONSUMER_SECRET': None,
}

"""
Set this to whereever the created CSV files should be placed (default is the same folder as this config file)
"""
CSV_DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
