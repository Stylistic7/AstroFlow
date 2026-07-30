# Cluster restart bump

Throwaway file to bump the cluster_version SHA, triggering the pull-deploy
workers (.3) to re-sync + restart their Ray — clearing a stale-cluster-ID
raylet on .3 that was spamming the head GCS. Safe to delete once .3 has
rejoined cleanly.
