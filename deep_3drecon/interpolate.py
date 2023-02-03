import tensorflow as tf


def interpolate_bilinear(
    grid,
    query_points,
    indexing = "ij",
    name = None,
):
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape
        `[batch, N, 2]`.
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
      name: a name for the operation (optional).
    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`
    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the
        inputs invalid.
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    with tf.name_scope(name or "interpolate_bilinear"):
        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)

        # grid shape checks
        grid_static_shape = grid.shape
        grid_shape = tf.shape(grid)
        if grid_static_shape.dims is not None:
            if len(grid_static_shape) != 4:
                raise ValueError("Grid must be 4D Tensor")
            if grid_static_shape[1] is not None and grid_static_shape[1] < 2:
                raise ValueError("Grid height must be at least 2.")
            if grid_static_shape[2] is not None and grid_static_shape[2] < 2:
                raise ValueError("Grid width must be at least 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        grid_shape[1], 2, message="Grid height must be at least 2."
                    ),
                    tf.debugging.assert_greater_equal(
                        grid_shape[2], 2, message="Grid width must be at least 2."
                    ),
                    tf.debugging.assert_less_equal(
                        tf.cast(
                            grid_shape[0] * grid_shape[1] * grid_shape[2],
                            dtype=tf.dtypes.float32,
                        ),
                        np.iinfo(np.int32).max / 8.0,
                        message="The image size or batch size is sufficiently "
                        "large that the linearized addresses used by "
                        "tf.gather may exceed the int32 limit.",
                    ),
                ]
            ):
                pass

        # query_points shape checks
        query_static_shape = query_points.shape
        query_shape = tf.shape(query_points)
        if query_static_shape.dims is not None:
            if len(query_static_shape) != 3:
                raise ValueError("Query points must be 3 dimensional.")
            query_hw = query_static_shape[2]
            if query_hw is not None and query_hw != 2:
                raise ValueError("Query points last dimension must be 2.")
        else:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        query_shape[2],
                        2,
                        message="Query points last dimension must be 2.",
                    )
                ]
            ):
                pass

        batch_size, height, width, channels = (
            grid_shape[0],
            grid_shape[1],
            grid_shape[2],
            grid_shape[3],
        )

        num_queries = query_shape[1]

        query_type = query_points.dtype
        grid_type = grid.dtype

        alphas = []
        floors = []
        ceils = []
        index_order = [0, 1] if indexing == "ij" else [1, 0]
        unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

        for i, dim in enumerate(index_order):
            with tf.name_scope("dim-" + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = grid_shape[i + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = tf.constant(0.0, dtype=query_type)
                floor = tf.math.minimum(
                    tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                )
                int_floor = tf.cast(floor, tf.dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = tf.cast(queries - floor, grid_type)
                min_alpha = tf.constant(0.0, dtype=grid_type)
                max_alpha = tf.constant(1.0, dtype=grid_type)
                alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = tf.expand_dims(alpha, 2)
                alphas.append(alpha)

            flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
            batch_offsets = tf.reshape(
                tf.range(batch_size) * height * width, [batch_size, 1]
            )

        # This wraps tf.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using tf.gather_nd.
        def gather(y_coords, x_coords, name):
            with tf.name_scope("gather-" + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = tf.gather(flattened_grid, linear_coordinates)
                return tf.reshape(gathered_values, [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], "top_left")
        top_right = gather(floors[0], ceils[1], "top_right")
        bottom_left = gather(ceils[0], floors[1], "bottom_left")
        bottom_right = gather(ceils[0], ceils[1], "bottom_right")

        # now, do the actual interpolation
        with tf.name_scope("interpolate"):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp