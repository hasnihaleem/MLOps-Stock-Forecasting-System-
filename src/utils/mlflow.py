import mlflow


def create_versioned_experiment(base_name="zoomcamps"):
    existing_experiments = mlflow.search_experiments()

    max_version = 0
    for exp in existing_experiments:
        if exp.name.startswith(base_name):
            suffix = exp.name[len(base_name) :]  # phần sau prefix
            if suffix.startswith("-") and suffix[1:].isdigit():
                version = int(suffix[1:])
                max_version = max(max_version, version)

    new_version = max_version + 1
    new_exp_name = f"{base_name}-{new_version}"

    # Artifact location trên S3
    artifact_location = f"s3://zoomcamps-bucket/mlflow/{new_exp_name}"

    # Tạo experiment
    experiment_id = mlflow.create_experiment(
        name=new_exp_name, artifact_location=artifact_location
    )

    print(f"✅ Created experiment: {new_exp_name} (id: {experiment_id})")
    return new_exp_name


def get_latest_versioned_experiment(base_name="zoomcamps"):
    existing_experiments = mlflow.search_experiments()

    max_version = -1
    latest_name = None

    for exp in existing_experiments:
        if exp.name.startswith(base_name):
            suffix = exp.name[len(base_name) :]
            if suffix.startswith("-") and suffix[1:].isdigit():
                version = int(suffix[1:])
                if version > max_version:
                    max_version = version
                    latest_name = exp.name

    return latest_name
