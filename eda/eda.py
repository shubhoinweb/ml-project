import numpy as np


def summarize_dataset(iris):
    data = iris.data
    targets = iris.target
    feature_names = list(iris.feature_names)

    species_names = [iris.target_names[target] for target in targets]
    species_counts = {}
    for species in species_names:
        species_counts[species] = species_counts.get(species, 0) + 1

    means_by_species = {}
    for species_idx, species_name in enumerate(iris.target_names):
        mask = targets == species_idx
        if np.any(mask):
            species_data = data[mask]
            means_by_species[species_name] = {
                feature: float(np.mean(species_data[:, idx]))
                for idx, feature in enumerate(feature_names)
            }

    summary = {
        "rows": len(data),
        "columns": len(feature_names),
        "species_counts": species_counts,
        "means": means_by_species,
    }

    return data, summary


def print_eda_summary(summary):
    print("\nEDA Summary")
    print("-" * 40)
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")
    print("Species counts:")
    for species, count in summary["species_counts"].items():
        print(f"  - {species}: {count}")

    print("\nMean feature values by species:")
    for species, values in summary["means"].items():
        print(f"  - {species}:")
        for feature, value in values.items():
            print(f"    {feature}: {value:.3f}")
