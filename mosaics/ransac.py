import numpy as np
import structlog
import itertools

logger = structlog.get_logger()
log = logger

ORTHOGONALITY_THRESHOLD = 0.05


def _find_all_possible_bases(centers):
  bases = []
  for combo in itertools.permutations(centers, 3):
    print('### combo', combo)
    origin = combo[0]
    print('### origin', origin)
    basis_vectors = np.asarray([combo[1] - origin, combo[2] - origin])
    print('### basis_vectors before normalizing', basis_vectors)
    basis_vectors = basis_vectors / np.linalg.norm(basis_vectors, axis=0)
    print('### basis_vectors after normalizing', basis_vectors)

    def flip_vector(v):
      if any(x < 0 for x in v):
        return -1 * v

      return v

    basis_vectors = np.vstack([flip_vector(v) for v in basis_vectors])
    print('### basis_vectors after flipping', basis_vectors)

    basis_dot = np.dot(basis_vectors[0], basis_vectors[1]) / (
        np.linalg.norm(basis_vectors[0]) * np.linalg.norm(basis_vectors[1]))
    if np.abs(basis_dot) <= ORTHOGONALITY_THRESHOLD:
      print('### found valid basis', basis_vectors)
      if any(
          np.array_equal(np.sort(b.flat), np.sort(basis_vectors.flat))
          for b in bases):
        print('### basis already in bases')
        continue

      bases.append(basis_vectors)

  print(f'### found {len(bases)} bases')
  print(bases)
  return bases


# A mosaic here is just a set of images that overlap at least a little. There may
# be no genuine mosaics here (e.g. none with any regular grid) or there may be
# several independent mosaics, each on their own independent grids.
#
# Maybe the first step is to look at every possible triplet of targets, and find
# all sets of orthogonal basis vectors?
#
# TODO: The definition of a mosaic is that the panes are aligned along their
# long or short axes. This means all their rotation_angles are consistent
# (at least for mosaics without rotation...).
def ransac(mosaic, debug=False, plot=False, plot_title=None):
  print(f'Running RANSAC on mosaic with {len(mosaic)} targets')

  if len(mosaic) == 1:
    print('### RANSAC called on a single target')
    return [t.stable_hash for t in mosaic], [(0, 0)]

  if len(mosaic) == 2:
    print('### RANSAC called with two targets')
    return [t.stable_hash for t in mosaic], [(0, 0), (0, 1)]

  best_error = np.inf
  best_origin = None
  best_inliers = []
  best_basis_vectors = None

  centers = np.vstack([t.center for t in mosaic]).T
  print(f'centers {centers}')

  for iteration in range(500):
    # print(f'Top of iteration {iteration}')

    # Choose 3 tiles at random. Compute two basis vectors from them.
    # Skip the iteration if the basis vectors aren't (nearly) orthogonal.
    # TODO: Does this really just always resolve to [[1, 0], [0, 1]]
    random_centers = centers[:,
                             np.random.
                             choice(centers.shape[1], 3, replace=False)]
    # print('random_centers', random_centers)
    origin = random_centers[:, 0]
    # print('origin', origin)
    basis_vectors = np.asarray(
        [random_centers[:, 1] - origin, random_centers[:, 2] - origin])
    # print('basis_vectors', basis_vectors)
    basis_dot = np.dot(basis_vectors[:, 0], basis_vectors[:, 1]) / (
        np.linalg.norm(basis_vectors[:, 0]) *
        np.linalg.norm(basis_vectors[:, 1]))
    # print('dot product of basis vectors', basis_dot)
    if np.abs(basis_dot) > 0.05:
      continue

    # Compute the error on all the centers, including only those that are very close
    # to integer coordinates.
    inliers = []
    errors = []
    for i, center in enumerate(centers.T):
      # print(f'  center {center}')
      transformed_center = np.linalg.solve(basis_vectors, center - origin)
      # print(f'  transformed center', transformed_center)
      error = np.linalg.norm(transformed_center - np.round(transformed_center))
      # print(f'  transformed_center {transformed_center} '
      #       f'has Euclidean error {error}')
      if error < 0.3:
        inliers.append(mosaic[i])
        errors.append(error)
    # print(f'Found {len(inliers)} inliers')

    if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers)
                                            and np.mean(error) < best_error):
      best_origin = origin
      best_inliers = inliers
      best_error = np.mean(errors)
      best_basis_vectors = basis_vectors
      # print('UPDATED BEST MODEL')

  # print('END OF RANSAC')
  # print('best_origin', best_origin)
  # print(f'best_inliers length {len(best_inliers)}: {best_inliers}')
  # print('best_error', best_error)
  # print('best_basis_vectors', best_basis_vectors)

  if not best_inliers:
    return [], []

  # print('centers in best basis:')
  all_tile_indices = []
  for i, center in enumerate(centers.T):
    transformed_center = np.linalg.solve(best_basis_vectors,
                                         center - best_origin)
    # print(f'  transformed_center {i}: {transformed_center}')
    error = np.linalg.norm(transformed_center - np.round(transformed_center))
    # print(f'  transformed_center {transformed_center} '
    #       f'has Euclidean error {error}')

    # TODO: Do these need to start at (0, 0) or is (-1, 1) acceptable, too?
    tile_indices = [int(x) for x in np.round(transformed_center)]
    print(f'  tile_indices {i}: {tile_indices}')
    all_tile_indices.append(tuple(tile_indices))

  target_ids = [t.stable_hash for t in best_inliers]
  print('### ransac returning target_ids', target_ids, 'all_tile_indices',
        all_tile_indices)
  return target_ids, all_tile_indices

  # plt.figure()
  # plt.title(plot_title)

  # # Draw the basis vectors.
  # for i, basis_vector in enumerate(best_basis_vectors.T):
  #   # print('plotting basis_vector', basis_vector)
  #   # print('best_origin[0]', best_origin[0])
  #   # print('best_origin[1]', best_origin[1])
  #   plt.plot([best_origin[0], best_origin[0] + basis_vector[0]],
  #            [best_origin[1], best_origin[1] + basis_vector[1]],
  #            label=f"Basis vector {i}",
  #            linewidth=5,
  #            color=f"C{i + 2}")

  # # Plot the centers of the N x M grid.
  # # points = []
  # # for x in range(np.max(integral_centers, axis=0)):
  # #   for y in range(np.max(integral_centers, axis=1)):
  # #     points.append(best_origin + x * best_basis_vectors[:, 0] +
  # #                   y * best_basis_vectors[:, 1])
  # # xs, ys = [p[0] for p in points], [p[1] for p in points]
  # # plt.scatter(xs, ys, color='red')

  # if plot:
  #   for target in best_inliers:
  #     plt.plot(*target.footprint_polygon().exterior.xy, color='black')
  #     plt.scatter([target.center[0]], [target.center[1]], color='black')

  #   outliers = [s for s in mosaic if s not in best_inliers]
  #   print(f'There are {len(outliers)} outliers')
  #   for target in outliers:
  #     plt.plot(*target.footprint_polygon().exterior.xy, color='#dddddd')
  #     plt.scatter([target.center[0]], [target.center[1]], color='#dddddd')

  #   plt.show()

  # return best_inliers
