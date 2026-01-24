from context import cache

image_hashes = cache["image_hashes"]
fits_headers = cache["fits_headers"]
plate_solutions = cache["plate_solutions"]

# hashes = ['3997920a96aff455', '3f2983b2163de3ea']
# hashes = [k for k, v in plate_solutions.items() if type(v) is str]
hashes = list(plate_solutions.keys())[0:100]

for hash in hashes:
    del image_hashes[hash]
    del fits_headers[hash]
    del plate_solutions[hash]

cache["image_hashes"] = image_hashes
cache["fits_headers"] = fits_headers
cache["plate_solutions"] = image_hashes
