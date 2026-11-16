"""Count image inputs for each decision."""


def temporal_image_count(data_config):
    image = data_config["temporal"]["image"]
    return int(data_config["num_camera_views"]) * (len(image.get("absolute_indices", [])) + len(image["delta_indices"]))
