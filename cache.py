import os
import diskcache

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'),
                        size_limit=int(50 * 1024 * 1024 * 1024),
                        eviction_policy='least-frequently-used')
