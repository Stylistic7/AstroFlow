import operator


def cluster_1d(points, epsilon=0.2, key=lambda x: x):
  clusters = []
  points_sorted = sorted(points, key=key)
  curr_point = points_sorted[0]
  curr_cluster = [curr_point]
  for point in points_sorted[1:]:
    if key(point) <= key(curr_point) + epsilon:
      curr_cluster.append(point)
    else:
      clusters.append(curr_cluster)
      curr_cluster = [point]
    curr_point = point
  clusters.append(curr_cluster)

  return clusters


if __name__ == '__main__':
  points = [(0.31, 'c'), (0.32, 'd'), (0.45, 'e'), (0.35, 'f'), (0.40, 'g'),
            (0.5, 'h'), (0.1, 'a'), (0.09, 'b')]
  clusters = cluster_1d(points, key=operator.itemgetter(0))
  for cluster in clusters:
    print(cluster)
