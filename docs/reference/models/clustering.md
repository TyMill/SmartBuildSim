# Clustering

`smartbuildsim.models.clustering` groups building zones using average sensor
values and scikit-learn's `KMeans`.

## Configuration

`ClusteringConfig` parameters:

- `sensors`: list of sensor names to include in the feature matrix.
- `n_clusters`: number of clusters to compute.
- `random_state`: deterministic initialisation for reproducible results.

## Workflow

- `_aggregate_features(data, sensors)` pivots the dataset to compute per-zone
  averages and validates that all sensors have readings for each zone.
- `cluster_zones(data, config)` standardises the features, fits `KMeans`, and
  returns a `ClusteringResult` containing assignments and the trained model.
- `ClusteringResult.describe_clusters()` exposes centroids as a tidy dataframe.

## CLI usage

```bash
smartbuildsim cluster run examples/configs/default.yaml
```

The command outputs `outputs/clusters.csv`, which pairs each zone with a cluster
label. The CSV can be passed to `smartbuildsim viz plot` to colour sensor curves
by cluster.

## Python example

```python
from smartbuildsim.data.generator import DataGeneratorConfig, generate_dataset
from smartbuildsim.models.clustering import ClusteringConfig, cluster_zones
from smartbuildsim.scenarios.presets import get_scenario

scenario = get_scenario("office-small")
dataset = generate_dataset(
    scenario.building, DataGeneratorConfig(**scenario.data.dict())
)
config = ClusteringConfig(**scenario.clustering.dict())
result = cluster_zones(dataset, config)

print(result.assignments)
print(result.describe_clusters())
```

`examples/scripts/run_example.py` executes these steps to provide cluster
context for the quickstart scenario.
