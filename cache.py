import os
import diskcache

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'),
                        size_limit=int(20 * 1024 * 1024 * 1024))
