def create_monai_adapter(monai_transforms):
    def adapter_func(batch):
        # Store metadata separately
        metadata = batch.pop("metadata", None)

        # Apply MONAI transforms to the batch
        try:
            transformed_batch = monai_transforms(batch)
        except Exception as e:
            print(f"Error in MONAI transform: {e}")
            # Return original batch if transform fails
            transformed_batch = batch

        # Handle the case where MONAI returns a list (e.g., from RandSpatialCropSamplesd)
        if isinstance(transformed_batch, list):
            print(
                f"Info: MONAI transform returned list with {len(transformed_batch)} samples, taking first one"
            )
            if len(transformed_batch) > 0:
                transformed_batch = transformed_batch[0]
            else:
                transformed_batch = batch

        # Ensure we return a dictionary
        if not isinstance(transformed_batch, dict):
            print(
                f"Warning: MONAI transform returned {type(transformed_batch)}, expected dict"
            )
            transformed_batch = batch

        # Restore metadata
        if metadata is not None:
            transformed_batch["metadata"] = metadata

        return transformed_batch

    return adapter_func
