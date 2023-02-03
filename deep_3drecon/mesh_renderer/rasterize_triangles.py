# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable triangle rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pathlib
import os
import tensorflow as tf

from . import camera_utils
# import camera_utils

file_path = pathlib.Path(__file__).parent.absolute()
rasterize_triangles_module = tf.load_op_library(
    #os.path.join(os.environ['TEST_SRCDIR'],
    os.path.join(file_path, 'kernels/rasterize_triangles_kernel.so'))


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
  """Rasterizes a mesh and computes interpolated vertex attributes.

  Applies projection matrices and then calls rasterize_clip_space().

  Args:
    world_space_vertices: 3-D float32 tensor of xyz positions with shape
      [batch_size, vertex_count, 3].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    camera_matrices: 3-D float tensor with shape [batch_size, 4, 4] containing
      model-view-perspective projection matrices.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  clip_space_vertices = camera_utils.transform_homogeneous(
      camera_matrices, world_space_vertices)
  return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                              image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
  """Rasterizes the input mesh expressed in clip-space (xyzw) coordinates.

  Interpolates vertex attributes using perspective-correct interpolation and
  clips triangles that lie outside the viewing frustum.

  Args:
    clip_space_vertices: 3-D float32 tensor of homogenous vertices (xyzw) with
      shape [batch_size, vertex_count, 4].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if len(clip_space_vertices.shape) != 3:
    raise ValueError('The vertex buffer must be 3D.')

  vertex_count = clip_space_vertices.shape[1].value

  batch_size = tf.shape(clip_space_vertices)[0]
  
  per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
    size=batch_size)
  per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

  def batch_loop_condition(b, *args):
    return b < batch_size

  def batch_loop_iteration(b, per_image_barycentric_coordinates,
    per_image_vertex_ids):
    barycentric_coords, triangle_ids, _ = (
        rasterize_triangles_module.rasterize_triangles(
            clip_space_vertices[b, :, :], triangles, image_width,
            image_height))
    per_image_barycentric_coordinates = \
      per_image_barycentric_coordinates.write(
        b, tf.reshape(barycentric_coords, [-1, 3]))

    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
    per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

    return b+1, per_image_barycentric_coordinates, per_image_vertex_ids

  b = tf.constant(0)
  _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
    batch_loop_condition, batch_loop_iteration,
    [b, per_image_barycentric_coordinates, per_image_vertex_ids])

  barycentric_coordinates = tf.reshape(
    per_image_barycentric_coordinates.stack(), [-1, 3])
  vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
  summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(summed_attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
  alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

  attributes_with_background = (
      alphas * attribute_images + (1.0 - alphas) * background_value)

  return attributes_with_background


@tf.RegisterGradient('RasterizeTriangles')
def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
  # Gradients are only supported for barycentric coordinates. Gradients for the
  # z-buffer are not currently implemented. If you need gradients w.r.t. z,
  # include z as a vertex attribute when calling rasterize_triangles.
  del df_dids, df_dz
  return rasterize_triangles_module.rasterize_triangles_grad(
      op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
      op.get_attr('image_width'), op.get_attr('image_height')), None
